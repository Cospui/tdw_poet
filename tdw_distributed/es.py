import numpy as np
import logging
import time
import json
import torch
import ray
from collections import namedtuple
from tdw_distributed.stats import compute_centered_ranks, batched_weighted_sum

from tdw_distributed.logger import CSVLogger
from tdw_distributed.noise_module import noise

logger = logging.getLogger(__name__)

# related to get_step
StepStats = namedtuple('StepStats', [
    'po_returns_mean',
    'po_returns_median',
    'po_returns_std',
    'po_returns_max',
    'po_theta_max',
    'po_returns_min',
    'po_len_mean',
    'po_len_std',
    'noise_std',
    'learning_rate',
    'theta_norm',
    'grad_norm',
    'update_ratio',
    'episodes_this_step',
    'timesteps_this_step',
    'time_elapsed_this_step',
])

EvalStats = namedtuple('StepStats', [
    'eval_returns_mean',
    'eval_returns_median',
    'eval_returns_std',
    'eval_len_mean',
    'eval_len_std',
    'eval_n_episodes',
    'time_elapsed',
])

# the return value of run_po_batch
POResult = namedtuple('POResult', [
    'noise_inds',
    'returns',
    'lengths',
])
EvalResult = namedtuple('EvalResult', ['returns', 'lengths'])


# @ray.remote
def initialize_worker():
    """define them as global variables so that their values can be modified in the function"""
    print("1")
    global niches, thetas, random_state, noise
    print("2")
    from .noise_module import noise
    print("3")
    import numpy as np
    print("4")
    random_state = np.random.RandomState()
    print("5")
    niches, thetas = {}, {}
    print("6")
    print("workers initialized")
    logger.info("initializing workers...")


def setup_worker(optim_id, niche):
    global niches
    niches[optim_id] = niche


def set_worker_theta(theta, optim_id):
    global thetas
    thetas[optim_id] = theta


def cleanup_worker(optim_id):
    global niches
    niches.pop(optim_id, None)


def add_env_to_niche(optim_id, env):
    global niches
    niches[optim_id].add_env(env)


def delete_env_from_niche(optim_id, env_name):
    global niches
    niches[optim_id].delete_env(env_name)


def add_noise_to_theta(theta, noise_std, noise_list=None):
    if noise_list is None:
        noise_list = []
        for param_tensor in theta:
            # print( param_tensor, "\t", theta[param_tensor].size(), "\t", theta[param_tensor] )
            tensor_shape = list(theta[param_tensor].size())
            # print("## ES -> add noise  @", param_tensor, " : ", tensor_shape)
            if param_tensor == "conv.weight":
                noise_layer = (np.random.rand(tensor_shape[0], tensor_shape[1], tensor_shape[2],
                                              tensor_shape[3]) - 0.5) * noise_std
                theta[param_tensor] += torch.from_numpy(noise_layer)
                noise_list.append(noise_layer)
            elif param_tensor == "conv.bias":
                noise_layer = (np.random.rand(tensor_shape[0]) - 0.5) * noise_std
                theta[param_tensor] += torch.from_numpy(noise_layer)
                noise_list.append(noise_layer)
            elif param_tensor == "l1.weight" or param_tensor == "l2.weight" or param_tensor == "l3.weight":
                noise_layer = (np.random.rand(tensor_shape[0], tensor_shape[1]) - 0.5) * noise_std
                theta[param_tensor] += torch.from_numpy(noise_layer)
                noise_list.append(noise_layer)
            elif param_tensor == "l1.bias" or param_tensor == "l2.bias" or param_tensor == "l3.bias":
                noise_layer = (np.random.rand(tensor_shape[0]) - 0.5) * noise_std
                theta[param_tensor] += torch.from_numpy(noise_layer)
                noise_list.append(noise_layer)

        return theta, np.asarray(noise_list)
    else:
        idx = 0
        for param_tensor in theta:
            # print( param_tensor, "\t", theta[param_tensor].size(), "\t", theta[param_tensor] )
            noise_layer = - noise_list[idx]
            idx += 1
            theta[param_tensor] += torch.from_numpy(noise_layer)
            # tensor_shape = list(theta[param_tensor].size())
            # if param_tensor == "conv.weight":
            #     noise_layer = noise_list[idx]
            #     idx += 1
            #     # print( "### ES -> add noise : ", noise )
            #     theta[param_tensor] += torch.from_numpy(noise_layer)
            # elif param_tensor == "conv.bias":
            #     noise_layer = noise_list[idx]
            #     idx += 1
            #     theta[param_tensor] += torch.from_numpy(noise_layer)
            # elif param_tensor == "l1.weight" or param_tensor == "l2.weight" or param_tensor == "l3.weight":
            #     noise_layer = noise_list[idx]
            #     idx += 1
            #     theta[param_tensor] += torch.from_numpy(noise_layer)
            # elif param_tensor == "l1.bias" or param_tensor == "l2.bias" or param_tensor == "l3.bias":
            #     noise_layer = noise_list[idx]
            #     idx += 1
            #     theta[param_tensor] += torch.from_numpy(noise_layer)
        return theta, None


def run_po_batch(optim_id, batch_size, rs_seed, noise_std, proposal=False):
    """
    +/- noise and created #batch_size theta to rollout & test
    return the returns & length
    """

    global noise, niches, thetas, random_state
    # get niche and theta
    niche = niches[optim_id]
    theta = thetas[optim_id]  # what i get here is a state_dict

    random_state.seed(rs_seed)

    ##TODO: I don't know if it's right about the noise_inds
    ##TODO: we need to implement the get length in the model

    returns = np.zeros((batch_size, 2))
    lengths = np.zeros((batch_size, 2), dtype='int')

    pos_thetas = []
    neg_thetas = []
    noise_list = []
    for _ in range(batch_size):
        pos_theta, noise_l = add_noise_to_theta(theta, noise_std, noise_list=None)
        pos_thetas.append(pos_theta)
        noise_list.append(noise_l)
    for i in range(batch_size):
        neg_theta, _ = add_noise_to_theta(theta, noise_std, noise_list=noise_list[i])
        neg_thetas.append(neg_theta)

    returns[:, 0], lengths[:, 0] = niche.rollout_batch(pos_thetas, batch_size, random_state, proposal=proposal)

    returns[:, 1], lengths[:, 1] = niche.rollout_batch(neg_thetas, batch_size, random_state, proposal=proposal)

    pores = POResult(returns=returns, noise_inds=noise_list, lengths=lengths)
    # print("### ES -> run po batch , get : ", pores)
    return pores


def run_eval_batch(optim_id, batch_size, rs_seed, noise_std=None, proposal=False):
    """
    eval_batch_size is 1
    eval batch
    same as training process, both of them use rollout_batch func
    we use exactly theta( rather than theta +- noise in training )
    and we return the eval result
    """
    global noise, niches, thetas, random_state
    niche = niches[optim_id]
    theta = thetas[optim_id]

    random_state.seed(rs_seed)

    returns, lengths = niche.rollout_batch((theta for i in range(batch_size)),
                                           batch_size, random_state, eval=True, proposal=proposal)

    return EvalResult(returns=returns, lengths=lengths)


class ESOptimizer:
    def __init__(self,
                 # engines,
                 # scheduler,
                 theta,
                 # make_niche,
                 niche,
                 learning_rate,
                 batches_per_chunk,
                 batch_size,
                 eval_batch_size,
                 eval_batches_per_step,
                 l2_coeff,
                 noise_std,
                 lr_decay=1,
                 lr_limit=0.001,
                 noise_decay=1,
                 noise_limit=0.01,
                 normalize_grads_by_noise_std=False,
                 returns_normalization='centered_ranks',
                 optim_id=0,
                 log_file='unname.log',
                 created_at=0,
                 is_candidate=False):

        from .optimizers import Adam, SimpleSGD

        logger.debug('Creating optimizer {}...'.format(optim_id))
        self.optim_id = optim_id
        # self.engines = engines
        # engines.block = True
        # self.scheduler = scheduler
        self.theta = theta
        # logger.debug('Optimizer {} optimizing {} parameters'.format(
        #     optim_id, len(theta)))

        # TODO: maybe change to pytorch version
        self.optimizer = Adam(self.theta, stepsize=learning_rate)
        self.sgd_optimizer = SimpleSGD(stepsize=learning_rate)

        self.lr_decay = lr_decay
        self.lr_limit = lr_limit
        self.noise_decay = noise_decay
        self.noise_limit = noise_limit

        logger.debug('Optimizer {} setting up 1 workers...'.format(
            optim_id))

        setup_worker(optim_id, niche)  # TODO: ray parallel
        # engines.apply(setup_worker, optim_id, make_niche) # niches [optim_id] = make_niche()

        self.batches_per_chunk = batches_per_chunk
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_batches_per_step = eval_batches_per_step
        self.l2_coeff = l2_coeff
        self.noise_std = noise_std
        self.init_noise_std = noise_std

        self.normalize_grads_by_noise_std = normalize_grads_by_noise_std
        self.returns_normalization = returns_normalization

        self.data_logger = None

        if not is_candidate:
            log_fields = [
                'po_returns_mean_{}'.format(optim_id),
                'po_returns_median_{}'.format(optim_id),
                'po_returns_std_{}'.format(optim_id),
                'po_returns_max_{}'.format(optim_id),
                'po_returns_min_{}'.format(optim_id),
                'po_len_mean_{}'.format(optim_id),
                'po_len_std_{}'.format(optim_id),
                'noise_std_{}'.format(optim_id),
                'learning_rate_{}'.format(optim_id),
                'eval_returns_mean_{}'.format(optim_id),
                'eval_returns_median_{}'.format(optim_id),
                'eval_returns_std_{}'.format(optim_id),
                'eval_len_mean_{}'.format(optim_id),
                'eval_len_std_{}'.format(optim_id),
                'eval_n_episodes_{}'.format(optim_id),
                'theta_norm_{}'.format(optim_id),
                'grad_norm_{}'.format(optim_id),
                'update_ratio_{}'.format(optim_id),
                'episodes_this_step_{}'.format(optim_id),
                'episodes_so_far_{}'.format(optim_id),
                'timesteps_this_step_{}'.format(optim_id),
                'timesteps_so_far_{}'.format(optim_id),
                'time_elapsed_this_step_{}'.format(optim_id),

                'accept_theta_in_{}'.format(optim_id),
                'eval_returns_mean_best_in_{}'.format(optim_id),
                'eval_returns_mean_best_with_ckpt_in_{}'.format(optim_id),
                'eval_returns_mean_theta_from_others_in_{}'.format(optim_id),
                'eval_returns_mean_proposal_from_others_in_{}'.format(optim_id),
            ]
            log_path = log_file + '/' + log_file.split('/')[-1] + '.' + str(optim_id) + '.log'

            logger.info('Optimizer {} created!'.format(optim_id))
            self.data_logger = CSVLogger(log_path, log_fields + [
                'time_elapsed_so_far',
                'iteration',
            ])

        self.filename_best = log_file + '/' + log_file.split('/')[-1] + '.' + optim_id + '.best.json'
        self.log_data = {}
        self.t_start = time.time()
        self.episodes_so_far = 0
        self.timesteps_so_far = 0

        self.checkpoint_thetas = None
        self.checkpoint_scores = None

        self.self_evals = None  # Score of current parent theta
        self.proposal = None  # Score of best transfer
        self.proposal_theta = None  # Theta of best transfer
        self.proposal_source = None  # Source of best transfer

        self.created_at = created_at
        self.start_score = None

        self.best_score = None
        self.best_theta = None

    # TODO!! Following codes are not seen

    def __del__(self):
        logger.debug('Optimizer {} cleanning up 1 workers...'.format(
            self.optim_id))
        # self.engines.apply(cleanup_worker, self.optim_id)
        cleanup_worker(self.optim_id)

    def clean_dicts_before_iter(self):
        # just clean the params
        self.log_data.clear()
        self.self_evals = None
        self.proposal = None
        self.proposal_theta = None
        self.proposal_source = None

    def pick_proposal(self, checkpointing, reset_optimizer):

        accept_key = 'accept_theta_in_{}'.format(
            self.optim_id)
        if checkpointing and self.checkpoint_scores > self.proposal:
            self.log_data[accept_key] = 'do_not_consider_CP'
        else:
            self.log_data[accept_key] = '{}'.format(
                self.proposal_source)
            if self.optim_id != self.proposal_source:
                # if the transfer is validate and we can make use its theta as this env's theta
                self.set_theta(
                    self.proposal_theta,
                    reset_optimizer=reset_optimizer)  # reset optimizer (SGD or Adam)
                self.self_evals = self.proposal
                print("#   ES -> pick proposal : ", self.proposal_source)

        self.checkpoint_thetas = self.theta
        self.checkpoint_scores = self.self_evals

        if self.best_score < self.self_evals:
            self.best_score = self.self_evals
            self.best_theta = self.theta

    def save_to_logger(self, iteration):
        self.log_data['time_elapsed_so_far'] = time.time() - self.t_start
        self.log_data['iteration'] = iteration
        if self.data_logger is not None:
            self.data_logger.log(**self.log_data)

        logger.debug('iter={} Optimizer {} best score {}'.format(
            iteration, self.optim_id, self.best_score))

        # if iteration % 100 == 0:
        #    self.save_policy(self.filename_best+'.arxiv.'+str(iteration))

        self.save_policy(self.filename_best)

    def save_policy(self, policy_file, reset=False):
        if self.best_score is not None and self.best_theta is not None:
            torch.save(self.best_theta, policy_file)
            # self.best_theta_list = []
            # try:
            #     for param_tensor in self.best_theta:
            #         self.best_theta_list.append(self.best_theta[param_tensor].numpy().tolist())
            # except Exception as e:
            #     print(e)
            #
            # with open(policy_file, 'wt') as out:
            #     json.dump([self.best_theta_list, self.best_score], out, sort_keys=True, indent=0,
            #               separators=(',', ': '))
            if reset:
                self.best_score = None
                self.best_theta = None

    def update_dicts_after_transfer(self, source_optim_id, source_optim_theta, stats, keyword):
        eval_key = 'eval_returns_mean_{}_from_others_in_{}'.format(keyword,  # noqa
                                                                   self.optim_id)
        if eval_key not in self.log_data.keys():
            self.log_data[eval_key] = source_optim_id + '_' + str(stats.eval_returns_mean)
        else:
            self.log_data[eval_key] += '_' + source_optim_id + '_' + str(stats.eval_returns_mean)

        if stats.eval_returns_mean > self.proposal:
            # keep track of the best score
            self.proposal = stats.eval_returns_mean
            self.proposal_source = source_optim_id + ('' if keyword == 'theta' else "_proposal")
            self.proposal_theta = source_optim_theta

    def update_dicts_after_es(self, stats, self_eval_stats):

        self.self_evals = self_eval_stats.eval_returns_mean
        if self.start_score is None:
            self.start_score = self.self_evals
        self.proposal = self_eval_stats.eval_returns_mean
        self.proposal_source = self.optim_id
        self.proposal_theta = self.theta

        if self.checkpoint_scores is None:
            self.checkpoint_thetas = self.theta
            self.checkpoint_scores = self_eval_stats.eval_returns_mean

        self.episodes_so_far += stats.episodes_this_step
        self.timesteps_so_far += stats.timesteps_this_step

        if self.best_score is None or self.best_score < self.self_evals:
            self.best_score = self.self_evals
            self.best_theta = self.theta

        self.log_data.update({
            'po_returns_mean_{}'.format(self.optim_id):
                stats.po_returns_mean,
            'po_returns_median_{}'.format(self.optim_id):
                stats.po_returns_median,
            'po_returns_std_{}'.format(self.optim_id):
                stats.po_returns_std,
            'po_returns_max_{}'.format(self.optim_id):
                stats.po_returns_max,
            'po_returns_min_{}'.format(self.optim_id):
                stats.po_returns_min,
            'po_len_mean_{}'.format(self.optim_id):
                stats.po_len_mean,
            'po_len_std_{}'.format(self.optim_id):
                stats.po_len_std,
            'noise_std_{}'.format(self.optim_id):
                stats.noise_std,
            'learning_rate_{}'.format(self.optim_id):
                stats.learning_rate,
            'eval_returns_mean_{}'.format(self.optim_id):
                self_eval_stats.eval_returns_mean,
            'eval_returns_median_{}'.format(self.optim_id):
                self_eval_stats.eval_returns_median,
            'eval_returns_std_{}'.format(self.optim_id):
                self_eval_stats.eval_returns_std,
            'eval_len_mean_{}'.format(self.optim_id):
                self_eval_stats.eval_len_mean,
            'eval_len_std_{}'.format(self.optim_id):
                self_eval_stats.eval_len_std,
            'eval_n_episodes_{}'.format(self.optim_id):
                self_eval_stats.eval_n_episodes,
            'theta_norm_{}'.format(self.optim_id):
                stats.theta_norm,
            'grad_norm_{}'.format(self.optim_id):
                stats.grad_norm,
            'update_ratio_{}'.format(self.optim_id):
                stats.update_ratio,
            'episodes_this_step_{}'.format(self.optim_id):
                stats.episodes_this_step,
            'episodes_so_far_{}'.format(self.optim_id):
                self.episodes_so_far,
            'timesteps_this_step_{}'.format(self.optim_id):
                stats.timesteps_this_step,
            'timesteps_so_far_{}'.format(self.optim_id):
                self.timesteps_so_far,
            'time_elapsed_this_step_{}'.format(self.optim_id):
                stats.time_elapsed_this_step + self_eval_stats.time_elapsed,
            'accept_theta_in_{}'.format(self.optim_id): 'self'
        })

    def broadcast_theta(self, theta):
        '''On all worker, set thetas[this optimizer] to theta'''
        logger.debug('Optimizer {} broadcasting theta...'.format(self.optim_id))
        set_worker_theta(theta, self.optim_id)
        # self.engines.apply(set_worker_theta, theta, self.optim_id) # thetas[optim_id] = theta

    def add_env(self, env):
        '''On all worker, add env_name to niche'''
        logger.debug('Optimizer {} add env {}...'.format(self.optim_id, env.name))
        # self.engines.apply(add_env_to_niche, self.optim_id, env)
        add_env_to_niche(self.optim_id, env)

    def delete_env(self, env_name):
        '''On all worker, delete env from niche'''
        logger.debug('Optimizer {} delete env {}...'.format(self.optim_id, env_name))
        # self.engines.apply(delete_env_from_niche, self.optim_id, env_name)
        delete_env_from_niche(self.optim_id, env_name)

    def start_chunk(self, runner, batches_per_chunk, batch_size, noise_std=None, proposal=False):
        # by calling the runner( run PO batch ), we get the results for tasks.
        # call batches_per_chunk times
        # logger.info('Optimizer {} spawning {} batches of size {}'.format(
        #     self.optim_id, batches_per_chunk, batch_size))
        rs_seeds = np.random.randint(np.int32(2 ** 31 - 1), size=batches_per_chunk)

        # chunk_tasks = []
        return runner(self.optim_id, batch_size, rs_seeds[0], noise_std=noise_std, proposal=proposal)
        # for i in range(1):
        #     chunk_tasks.append(
        #         self.scheduler.apply(runner, self.optim_id, batch_size, rs_seeds[i], *args))

        # return chunk_tasks

    def get_chunk(self, tasks):

        # print('-------get_chunk-------')
        # from IPython import embed; embed()

        return [task.get() for task in tasks]

    def collect_po_results(self, po_results):
        # print( "in collect po results", po_results )
        # print("### ES -> collect po res : ", po_results)
        # noise_inds = np.concatenate([r.noise_inds for r in po_results])
        # returns = np.concatenate([r.returns for r in po_results])
        # lengths = np.concatenate([r.lengths for r in po_results])
        noise_inds = np.concatenate([po_results.noise_inds])
        returns = np.concatenate([po_results.returns])
        lengths = np.concatenate([po_results.lengths])
        return noise_inds, returns, lengths

    def collect_eval_results(self, eval_results):
        # just concatenate
        # eval_returns = np.concatenate([r.returns for r in eval_results])
        # eval_lengths = np.concatenate([r.lengths for r in eval_results])
        # print("### ES -> collect eval res : ", eval_results)

        eval_returns = np.concatenate([eval_results.returns])
        eval_lengths = np.concatenate([eval_results.lengths])

        return eval_returns, eval_lengths

    def compute_grads(self, step_results, theta):
        """
        :param step_results:
        :param theta:
        :return: SUM(E*noise)/(n*std) and the best Theta
        """

        noise_inds, returns, _ = self.collect_po_results(step_results)

        pos_row, neg_row = returns.argmax(
            axis=0)  # get the best result from + noise / - noise, store the idx of row in pos_row & neg_row
        noise_sign = 1.0
        po_noise_ind_max = noise_inds[pos_row]  # from the idx to find the noise itself

        if returns[pos_row, 0] < returns[neg_row, 1]:
            noise_sign = -1.0
            po_noise_ind_max = noise_inds[neg_row]  # from the idx to find the noise itself

        i = 0
        for param_tensor in theta:
            theta[param_tensor] += torch.from_numpy(po_noise_ind_max[i])
            i += 1

        proc_returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        grads, _ = batched_weighted_sum(proc_returns[:, 0] - proc_returns[:, 1],
                                        (noise_inds[idx] for idx in range(len(noise_inds))),
                                        batch_size=self.batch_size)
        grads /= len(returns)
        if self.normalize_grads_by_noise_std:
            grads /= self.noise_std

        return grads, theta

    def set_theta(self, theta, reset_optimizer=True):
        self.theta = theta
        if reset_optimizer:
            self.optimizer.reset()
            self.noise_std = self.init_noise_std

    def start_theta_eval(self, theta):
        """
            eval theta in this optimizer's niche
        """
        step_t_start = time.time()
        self.broadcast_theta(theta)  # parallel related, set theta to all workers.

        eval_tasks = self.start_chunk(run_eval_batch,
                                      self.eval_batches_per_step,
                                      self.eval_batch_size)
        return eval_tasks, theta, step_t_start

    def get_theta_eval(self, res):
        """
            get the result of evaluation
        """
        eval_tasks, theta, step_t_start = res
        eval_results = eval_tasks
        # eval_results = self.get_chunk(eval_tasks) # same as get_step, a list of scheduler.apply.get()
        eval_returns, eval_lengths = self.collect_eval_results(eval_results)  # just concatenate
        step_t_end = time.time()

        # logger.debug(
        #     'get_theta_eval {} finished running {} episodes, {} timesteps'.format(
        #         self.optim_id, len(eval_returns), eval_lengths.sum()))

        print("# ES -> get eval res: ", eval_returns)

        return EvalStats(eval_returns_mean=eval_returns.mean(),
                         eval_returns_median=np.median(eval_returns),
                         eval_returns_std=eval_returns.std(),
                         eval_len_mean=eval_lengths.mean(),
                         eval_len_std=eval_lengths.std(),
                         eval_n_episodes=len(eval_returns),
                         time_elapsed=step_t_end - step_t_start)

    def start_step(self, theta=None, proposal=False):
        """
            based on theta (if none, this optimizer's theta)
            generate the P.O. cloud, and eval them in this optimizer's niche
        """
        step_t_start = time.time()
        if theta is None:
            theta = self.theta

        self.broadcast_theta(theta)
        step_results = self.start_chunk(run_po_batch,  # runner
                                        self.batches_per_chunk,
                                        self.batch_size,
                                        self.noise_std,
                                        proposal=proposal)
        return step_results, theta, step_t_start

    def get_step(self, res, propose_with_adam=True, decay_noise=True, propose_only=False):
        """
        :param res:
        :param propose_with_adam:
        :param decay_noise:
        :param propose_only: True when transfer / eval_transfer
        :return: optimized theta (after a single ES step)
        """

        step_results, theta, step_t_start = res
        # step_results = self.get_chunk(step_tasks) # TODO ray parallel

        _, po_returns, po_lengths = self.collect_po_results(step_results)  # concatenate
        episodes_this_step = len(po_returns)  # NOTE: I think this means the reward in TDW
        timesteps_this_step = po_lengths.sum()  # NOTE: I think this means the time spent in TDW

        logger.info('Optimizer {} finished running {} episodes, {} timesteps'.format
                    (self.optim_id, episodes_this_step, timesteps_this_step))

        grads, po_theta_max = self.compute_grads(step_results, theta)  # get the grads and best THETA
        if not propose_only:
            # ratio is the norm of step / norm of step, theta is the results of (theta + optimization)
            update_ratio, theta = self.optimizer.update(
                theta, -grads, self.l2_coeff)  # + self.l2 * theta

            self.optimizer.stepsize = max(
                self.optimizer.stepsize * self.lr_decay, self.lr_limit)  # update the stepsize after each step
            if decay_noise:
                self.noise_std = max(
                    self.noise_std * self.noise_decay, self.noise_limit)
            print("### ES -> get step, update theta for it's NOT propose only")
        else:  # only make proposal
            # actually do not change the params, one step optimization
            if propose_with_adam:
                update_ratio, theta = self.optimizer.propose(
                    theta, -grads, self.l2_coeff)
            else:
                update_ratio, theta = self.sgd_optimizer.compute(
                    theta, -grads, self.l2_coeff)  # keeps no state

            print("### ES -> get step, update theta for it's propose ONLY")

        logger.info(
            'Optimizer {} finished computing gradients'.format(
                self.optim_id))

        step_t_end = time.time()

        # print("### ES -> get step done ")

        return theta, StepStats(po_returns_mean=po_returns.mean(),
                                po_returns_median=np.median(po_returns),
                                po_returns_std=po_returns.std(),
                                po_returns_max=po_returns.max(),
                                po_theta_max=po_theta_max,
                                po_returns_min=po_returns.min(),
                                po_len_mean=po_lengths.mean(),
                                po_len_std=po_lengths.std(),
                                noise_std=self.noise_std,
                                learning_rate=self.optimizer.stepsize,
                                theta_norm=250,  # completely useless
                                grad_norm=250,  # completely useless
                                update_ratio=250.0,  # useless
                                episodes_this_step=episodes_this_step,
                                timesteps_this_step=timesteps_this_step,
                                time_elapsed_this_step=step_t_end - step_t_start,
                                )

    def evaluate_theta(self, theta):
        self_eval_task = self.start_theta_eval(theta)
        self_eval_stats = self.get_theta_eval(self_eval_task)
        return self_eval_stats.eval_returns_mean

    def evaluate_transfer(self, optimizers, propose_with_adam=False):
        '''
        used when generate new env from mutation
        the class itself is an optimizers, which contains an env and a theta
        get thetas from other optimizers, and try them on the env to see whether it's more suitable
        '''

        best_init_score = None
        best_init_theta = None

        for source_optim in optimizers.values():
            # directly use the theta as parameter

            score = self.evaluate_theta(source_optim.theta)
            print("### ES -> eval transfer @ directly : ", score)
            if best_init_score is None or score > best_init_score:
                best_init_score = score
                best_init_theta = source_optim.theta

            # use theta as parameter after an ES step
            task = self.start_step(source_optim.theta)  # task stands for: step_results, theta, step_t_start
            proposed_theta, _ = self.get_step(
                task, propose_with_adam=propose_with_adam, propose_only=True)
            score = self.evaluate_theta(proposed_theta)
            print("                        @ propose : ", score)
            if score > best_init_score:
                best_init_score = score
                best_init_theta = proposed_theta

        return best_init_score, best_init_theta
