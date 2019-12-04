import logging

logger = logging.getLogger(__name__)
import json
import numpy as np
from collections import OrderedDict, namedtuple

from tdw_distributed.niches.tdwniches import Env_config
from tdw_distributed.es import ESOptimizer, initialize_worker
from tdw_distributed.novelty import compute_novelty_vs_archive
from tdw_distributed.reproduce_ops import Reproducer


def construct_niche_fns_from_env(args, env, seed, is_candidate, new_child):
    """:return name and niche wrapper"""
    def niche_wrapper(configs, seed, is_candidate, new_child):
        def make_niche(): # in order not to create a new environment instantly
            from tdw_distributed.niches.tdwniches import TdwNiche
            return TdwNiche(env_configs=configs,
                            seed=seed,
                            args=args,
                            init=args.init,
                            stochastic=args.stochastic,
                            is_candidate=is_candidate,
                            new_child=new_child)

        return make_niche

    niche_name = env.name
    configs = (env,)

    return niche_name, niche_wrapper(list(configs), seed, is_candidate, new_child)


class MultiESOptimizer:
    def __init__(self, args):

        self.args = args
        self.env_registry = OrderedDict()
        self.env_archive = OrderedDict()
        self.env_reproducer = Reproducer(args)
        self.optimizers = OrderedDict()

        # TODO: not have been checked
        if args.start_from:
            logger.debug("args.start_from {}".format(args.start_from))
            with open(args.start_from) as f:
                start_from_config = json.load(f)

            logger.debug(start_from_config['path'])
            logger.debug(start_from_config['niches'])
            logger.debug(start_from_config['exp_name'])

            path = start_from_config['path']
            exp_name = start_from_config['exp_name']
            prefix = path + exp_name + '/' + exp_name + '.'
            for niche_name, niche_file in sorted(start_from_config['niches'].items()):
                logger.debug(niche_name)
                niche_file_complete = prefix + niche_file
                logger.debug(niche_file_complete)
                with open(niche_file_complete) as f:
                    data = json.load(f)
                    logger.debug('loading file %s' % (niche_file_complete))
                    model_params = np.array(data[0])  # assuming other stuff is in data
                    logger.debug(model_params)

                env_def_file = prefix + niche_name + '.env.json'
                with open(env_def_file, 'r') as f:
                    exp = json.loads(f.read())

                env = Env_config(**exp['config'])
                logger.debug(env)
                seed = exp['seed']
                self.add_optimizer(env=env, seed=seed, model_params=model_params)
        else:
            # create a brand-new env with the easiest settings.
            env = Env_config(name='trivial',
                             is_main_sc=False,
                             no_target=1,
                             no_cube_stack_target=0,
                             no_cones_target=0,
                             no_walled_target=0,
                             no_cube=0,
                             no_rectangles=0,
                             is_ramp_inside=False)

            self.add_optimizer(env=env, seed=args.master_seed)

    def create_optimizer(self, env, seed, created_at=0, model_params=None, is_candidate=False, new_child=False):
        """
            Return an ESOptimizer
            1- create a tdw-niche env
            2- use it as param and make an ESOptimizer
        """
        assert env is not None

        # optim_id is env.name, niche_fn is func to create a Tdw Niche
        optim_id, niche_fn = construct_niche_fns_from_env(args=self.args, env=env, seed=seed, is_candidate=is_candidate, new_child=new_child)

        niche = niche_fn()  # get a instance of TdwNiche
        if model_params is not None:
            theta = model_params
        else:
            theta = niche.initial_theta()
        assert optim_id not in self.optimizers.keys()  # the name of env, so that we can add it to the OrderDict later

        print("# POET_ALGO  -> create optimizer < ", optim_id, " > .")

        # TODO: ESOptimizer
        return ESOptimizer(optim_id=optim_id,
                           theta=theta,  # get theta
                           # make_niche=niche_fn,  # the the func to make niche
                           niche=niche,
                           learning_rate=self.args.learning_rate,
                           lr_decay=self.args.lr_decay,
                           lr_limit=self.args.lr_limit,
                           batches_per_chunk=self.args.batches_per_chunk,
                           batch_size=self.args.batch_size,
                           eval_batch_size=self.args.eval_batch_size,
                           eval_batches_per_step=self.args.eval_batches_per_step,
                           l2_coeff=self.args.l2_coeff,
                           noise_std=self.args.noise_std,
                           noise_decay=self.args.noise_decay,
                           normalize_grads_by_noise_std=self.args.normalize_grads_by_noise_std,
                           returns_normalization=self.args.returns_normalization,
                           noise_limit=self.args.noise_limit,
                           log_file=self.args.log_file,
                           created_at=created_at,
                           is_candidate=is_candidate)

    def add_optimizer(self, env, seed, created_at=0, model_params=None, new_child=False, created_optimizer=None):
        """
            create a new optimizer/niche according to env
            created_at: the iteration when this niche is created
        """
        if created_optimizer is None:
            o = self.create_optimizer(env, seed, created_at, model_params, new_child=new_child)
        else:
            o = created_optimizer
        optim_id = o.optim_id  # name of env
        self.optimizers[optim_id] = o  # add ESOptimizer to the OrderedDict

        assert optim_id not in self.env_registry.keys()
        assert optim_id not in self.env_archive.keys()
        self.env_registry[optim_id] = env  # add env to the env related params.
        self.env_archive[optim_id] = env

        print("# P_ALGO  -> add optimizer < ", optim_id, " > .")

        # dump the env
        log_file = self.args.log_file
        env_config_file = log_file + '/' + log_file.split('/')[-1] + '.' + optim_id + '.env.json'
        record = {'config': env._asdict(), 'seed': seed}
        with open(env_config_file, 'w') as f:
            json.dump(record, f)

    def delete_optimizer(self, optim_id):
        assert optim_id in self.optimizers.keys()
        # assume optim_id == env_id for single_env niches
        o = self.optimizers.pop(optim_id)
        del o
        assert optim_id in self.env_registry.keys()
        self.env_registry.pop(optim_id)
        logger.info('DELETED {} '.format(optim_id))

    def clean_up_ipyparallel(self):
        pass
        # logger.debug('Clean up ipyparallel ...')
        # self.client.purge_everything()
        # self.client.purge_results("all")
        # self.client.purge_local_results("all")
        # self.client.results.clear()
        # self.client.metadata.clear()
        # self.client._futures.clear()
        # self.client._output_futures.clear()

        # self.client.purge_hub_results("all")
        # self.client.history = []
        # self.client.session.digest_history.clear()

        # self.engines.results.clear()
        # self.scheduler.results.clear()
        # self.client.results.clear()
        # self.client.metadata.clear()

    def ind_es_step(self, iteration):
        """run a single ES step (update theta) and get the result of eval -> update the dict"""
        print("## P_ALGO -> ind es step @", iteration, "-> START")

        tasks = [o.start_step() for o in self.optimizers.values()]  # generate P.O. cloud && return their results
        # print("## P_ALGO -> total_length : ", len(tasks))
        for optimizer, task in zip(self.optimizers.values(), tasks):
            optimizer.theta, stats = optimizer.get_step(task)
            # asign the theta after optimization to optimizer after a single ES step
            self_eval_task = optimizer.start_theta_eval(optimizer.theta)  # evaluate
            self_eval_stats = optimizer.get_theta_eval(self_eval_task)  # get the result

            logger.info('Iter={} Optimizer {} theta_mean {} best po {} iteration spent {}'.format(
                iteration, optimizer.optim_id, self_eval_stats.eval_returns_mean,
                stats.po_returns_max, iteration - optimizer.created_at))

            # update optimizer result
            optimizer.update_dicts_after_es(stats=stats, self_eval_stats=self_eval_stats)

        self.clean_up_ipyparallel()
        print("## P_ALGO -> ind es step -> END")

    def transfer(self, propose_with_adam, checkpointing, reset_optimizer):
        # test from other env's solution, both original ones and proposal ones( one step optimization )
        # pick the best and replace
        # logger.info('Computing direct transfers...')
        print("## P_ALGO -> transfer : direct transfer ")
        for source_optim in self.optimizers.values():
            source_tasks = []
            # start and get pattern
            for target_optim in [o for o in self.optimizers.values()
                                 if o is not source_optim]:
                task = target_optim.start_theta_eval(source_optim.theta)
                source_tasks.append((task, target_optim))

            for task, target_optim in source_tasks:
                stats = target_optim.get_theta_eval(task)

                print("Transfer ", source_optim.optim_id, " -> ", target_optim.optim_id, " :: ", stats.eval_returns_mean)

                target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                                                         source_optim_theta=source_optim.theta,
                                                         stats=stats,
                                                         keyword='theta')

        # logger.info('Computing proposal transfers...')
        print("## P_ALGO -> transfer : proposal transfer ")
        for source_optim in self.optimizers.values():
            source_tasks = []
            for target_optim in [o for o in self.optimizers.values()
                                 if o is not source_optim]:
                task = target_optim.start_step(source_optim.theta, proposal=True)
                source_tasks.append((task, target_optim))

            for task, target_optim in source_tasks:
                proposed_theta, _ = target_optim.get_step(
                    task, propose_with_adam=propose_with_adam, propose_only=True)

                proposal_eval_task = target_optim.start_theta_eval(proposed_theta)
                proposal_eval_stats = target_optim.get_theta_eval(proposal_eval_task)

                target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                                                         source_optim_theta=proposed_theta,
                                                         stats=proposal_eval_stats,
                                                         keyword='proposal')

        logger.info('Considering transfers...')
        for o in self.optimizers.values():
            o.pick_proposal(checkpointing, reset_optimizer)  # just pick the best theta.

        self.clean_up_ipyparallel()

    def check_optimizer_status(self):
        '''
            take all the envs from env_registry whose self_evals > threshold.
            delete_candidates = []
        '''
        logger.info("health_check")
        repro_candidates, delete_candidates = [], []
        for optim_id in self.env_registry.keys():
            o = self.optimizers[optim_id]
            logger.info("niche {} created at {} start_score {} current_self_evals {}".format(
                optim_id, o.created_at, o.start_score, o.self_evals))
            if o.self_evals >= self.args.repro_threshold:  # self_evals is the average eval score
                repro_candidates.append(optim_id)

        logger.debug("candidates to reproduce")
        logger.debug(repro_candidates)
        logger.debug("candidates to delete")
        logger.debug(delete_candidates)

        return repro_candidates, delete_candidates

    def pass_dedup(self, env_config):
        """
            de-duplication
        """
        if env_config.name in self.env_registry.keys():
            logger.debug("active env already. reject!")
            return False
        else:
            return True

    def pass_mc(self, score):
        """
            make sure that the score fit the bound
        """
        if score < self.args.mc_lower or score > self.args.mc_upper:
            return False
        else:
            return True

    def get_new_env(self, list_repro):
        """
            randomly pick one from the list_repro
            call mutate in Reproducer to mutate it, and generate a child env
        """

        optim_id = self.env_reproducer.pick(list_repro)  # randomly pick one
        assert optim_id in self.optimizers.keys()
        assert optim_id in self.env_registry.keys()
        parent = self.env_registry[optim_id]
        child_env_config = self.env_reproducer.mutate(parent)  # from parent(env), mutate and get a new child(env)

        logger.info("we pick to mutate: {} and we got {} back".format(optim_id, child_env_config.name))
        logger.debug("parent")
        logger.debug(parent)
        logger.debug("child")
        logger.debug(child_env_config)

        seed = np.random.randint(1000000)
        return child_env_config, seed, optim_id

    def get_child_list(self, parent_list, max_children):
        """
        parent is a list of envs which evals > threshold
        we randomly pick one from parent list, and mutate it to get a child.
        de-duplication & test for the eval-score
        sorted by the novelty( with env_archive )
        """
        child_list = []

        mutation_trial = 0
        while mutation_trial < max_children:
            new_env_config, seed, parent_optim_id = self.get_new_env(parent_list)  # get one child env
            mutation_trial += 1
            if self.pass_dedup(new_env_config):  # de-duplication
                print("## P_ALGO -> get child list, new env pass dedup: ", new_env_config)
                # from IPython import embed
                # embed()

                o = self.create_optimizer(new_env_config, seed, is_candidate=True, new_child=True)
                score = o.evaluate_theta(self.optimizers[parent_optim_id].theta)  # eval it with parent's parameter
                # del o
                if self.pass_mc(score):  # make sure the env is not too simple or too difficult
                    novelty_score = compute_novelty_vs_archive(self.env_archive, new_env_config, k=5)
                    logger.info("{} passed mc, novelty score {}".format(score, novelty_score))
                    child_list.append((new_env_config, seed, parent_optim_id, novelty_score, o))
                else:
                    del o

        # sort child list according to novelty for high to low
        child_list = sorted(child_list, key=lambda x: x[3], reverse=True)
        return child_list

    def adjust_envs_niches(self, iteration, steps_before_adjust, max_num_envs=None, max_children=2, max_admitted=1):
        '''
        put the high score env to parent_list and get child list from them
        select the most suitable parameter for child env
        (remove the oldest env, maybe)
        '''

        if iteration > 0 and iteration % steps_before_adjust == 0:

            print("## P_ALGO adjust env niches ")

            # from IPython import embed
            # embed()

            list_repro, list_delete = self.check_optimizer_status()
            # all the env in env_registry whose self_eval > threshold 
            # list _delete is [](emtpy) and repro is candidate.self_evals that > threshold

            if len(list_repro) == 0:
                print(" FAIL #1 ")
                return
            else:
                print()

            logger.info("list of niches to reproduce")
            logger.info(list_repro)
            logger.info("list of niches to delete")
            logger.info(list_delete)

            child_list = self.get_child_list(list_repro, max_children)  # mutate to get children

            logger.info("children list")
            logger.info(child_list)

            # from IPython import embed
            # embed()

            if child_list == None or len(child_list) == 0:
                logger.info("mutation to reproduce env FAILED!!!")
                print(" FAIL #2 ")
                return

            admitted = 0
            for child in child_list:
                new_env_config, seed, _, _, o = child
                print("## P_ALGO -> child env: ", new_env_config)
                # targeted transfer
                # o = self.create_optimizer(new_env_config, seed, is_candidate=True, new_child=True)
                score_child, theta_child = o.evaluate_transfer(self.optimizers)

                print("## P_ALGO -> adjust env niches get score @", score_child)
                # from IPython import embed
                # embed()
                # del o
                if self.pass_mc(score_child):  # check mc
                    print("##### P_ALGO -> pass all checks, +", new_env_config.name)
                    self.add_optimizer(env=new_env_config,
                                       seed=seed,
                                       created_at=iteration,
                                       model_params=theta_child,
                                       new_child=True,
                                       created_optimizer=o)
                    admitted += 1
                    if admitted >= max_admitted:
                        break
                else:
                    del o

            if max_num_envs and len(self.optimizers) > max_num_envs:
                num_removals = len(self.optimizers) - max_num_envs
                self.remove_oldest(num_removals)

    def remove_oldest(self, num_removals):
        print("## P_ALGO -> remove # ", num_removals, " oldests ")
        # remove # of the oldest envs
        list_delete = []
        for optim_id in self.env_registry.keys():
            if len(list_delete) < num_removals:
                list_delete.append(optim_id)
            else:
                break

        for optim_id in list_delete:
            self.delete_optimizer(optim_id)

    def optimize(self,
                 iterations=200,
                 steps_before_transfer=25,
                 propose_with_adam=False,
                 checkpointing=False,
                 reset_optimizer=True):

        for iteration in range(iterations):
            # adjust the env( mutate and generate new ones while remove old ones )
            # TODO see later
            self.adjust_envs_niches(iteration,
                                    self.args.adjust_interval * steps_before_transfer,
                                    max_num_envs=self.args.max_num_envs)  # execute every certain steps

            print("## P_ALGO : optimize iter # ", iteration)
            # from IPython import embed
            # embed()

            for o in self.optimizers.values():  # value-key pair, and o is ESOptimizer
                o.clean_dicts_before_iter()  # just clean the parameters

            self.ind_es_step(iteration=iteration)  # a single ES step and update theta

            if len(self.optimizers) > 1 and iteration % steps_before_transfer == 0:
                print("## P_ALGO -> iter # ", iteration, " now in transfer")
                # from IPython import embed
                # embed()
                # test whether other thetas are better than origin ones
                self.transfer(propose_with_adam=propose_with_adam,
                              checkpointing=checkpointing,
                              reset_optimizer=reset_optimizer)

            if iteration % steps_before_transfer == 0:
                for o in self.optimizers.values():
                    o.save_to_logger(iteration)
