from collections import namedtuple, OrderedDict
from tdw_distributed.niches.model import simulate, Model, tdw_custom

Env_config = namedtuple('Env_config', [
    'name',
    'is_main_sc',
    'no_target', 'no_cube_stack_target', 'no_cones_target', 'no_walled_target',  # 4 types of targets
    'no_cube', 'no_rectangles',  # 3 types of obstacles
    'is_ramp_inside'  # can be True only when no_walled_target >0
])

DEFAULT_ENV = Env_config(name='trivial',
                         is_main_sc=False,
                         no_target=1,
                         no_cube_stack_target=0,
                         no_cones_target=0,
                         no_walled_target=0,
                         no_cube=0,
                         no_rectangles=0,
                         is_ramp_inside=False)


class TdwNiche:
    def __init__(self, env_configs, seed, args, init='random', stochastic=False, is_candidate=False, new_child=False):
        """
        :param env_configs: env_configs is a list of env config
        :param seed:
        :param args:
        :param init: how the parameter is initialized (can ONLY set to be random)
        :param stochastic:
        """

        self.args = args
        self.model = Model(tdw_custom)
        if not isinstance(env_configs, list):
            env_configs = [env_configs]
        self.env_configs = OrderedDict()
        for env in env_configs:
            self.env_configs[env.name] = env
        self.seed = seed
        self.stochastic = stochastic
        if not is_candidate:
            if not new_child:
                self.model.make_env(seed=seed, env_config=DEFAULT_ENV)
        self.init = init

        print("## Tdw Niches -> initialized ")

    def add_env(self, env):
        env_name = env.name
        assert env_name not in self.env_configs.keys()
        self.env_configs[env_name] = env

    def delete_env(self, env_name):
        assert env_name in self.env_configs.keys()
        self.env_configs.pop(env_name)

    def initial_theta(self):
        # initial the theta ( parameter )
        if self.init == 'random':
            return self.model.get_random_model_params()
        else:
            raise NotImplementedError(
                'Undefined initialization scheme `{}`'.format(self.init))

    def rollout(self, theta, random_state, eval=False, proposal=False):
        """
            by calling simulate in (.model)
            we can get the result of a batch's run
        """
        # TODO: what is th returns in our envs???
        # NOTE: maybe returns is the reward and lengths is time??
        # print( "###### TDW NICHES -> rollout : ", theta )
        self.model.set_model_params(theta)
        total_returns = 0
        total_length = 0
        # set the seed
        if self.stochastic:
            seed = random_state.randint(1000000)
        else:
            seed = self.seed

        for env_config in self.env_configs.values():
            # print("### TDW NICHES -> rollout : ", env_config)

            returns, lengths = simulate(self.model, seed=seed, train_mode=not eval, num_episode=1,
                                        env_config_this_sim=env_config)

            if (not eval) and (not proposal):
                print("### TDW NICHES -> rollout @", env_config.name, " get res: ", returns)

            return_sum = 0
            length_sum = 0
            return_len = 0
            length_len = 0
            for r in returns:
                return_sum += r
                return_len += 1
            for l in lengths:
                length_sum += l
                length_len += 1

            total_returns += return_sum / return_len
            total_length += length_sum / length_len
        return total_returns / len(self.env_configs), total_length

    def rollout_batch(self, thetas, batch_size, random_state, eval=False, proposal=False):
        """
            call rollout for each component in the batch, and return the list of each component
            NOTE batch_size equals len(thetas)
        """
        import numpy as np
        returns = np.zeros(batch_size)
        lengths = np.zeros(batch_size, dtype='int')
        if (not eval) and (not proposal):
            print('--------rollout_batch START--------')
        for i, theta in enumerate(thetas):
            # i is 0,1,2,3...
            returns[i], lengths[i] = self.rollout(theta, random_state=random_state, eval=eval, proposal=proposal)

        if (not eval) and (not proposal):
            print('--------rollout_batch END--------')
            print("#### TDW NICHES -> rollout batch get returns: ", returns, " lengths ", lengths)

        return returns, lengths
