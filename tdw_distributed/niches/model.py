import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from PIL import Image

from collections import namedtuple

import gym
# from gym_tdw rather then  tdw_gym.gym_tdw can solve the bug of re-register
from gym_tdw.envs.utils.proc_gen import create_puzzle, display_table, create_puzzle_poet_TASK1

logger = logging.getLogger(__name__)

Game = namedtuple('Game', ['env_name', 'time_factor', 'max_object',
                           'output_size', 'layers', 'noise_bias', 'output_noise'])

tdw_custom = Game(env_name="tdw_env",
                  max_object=22,
                  output_size=9,
                  time_factor=0,
                  layers=[128, 64],
                  noise_bias=0.0,
                  output_noise=[False, False, False])


class Policy(nn.Module):
    def __init__(self, max_object, unit_1, unit_2):
        # unit_1, unit_2 is the shape of linear layer
        super(Policy, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 4))
        self.l1 = nn.Linear(max_object, unit_1)
        self.l2 = nn.Linear(unit_1, unit_2)
        self.l3 = nn.Linear(unit_2, 9)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.squeeze(0).squeeze(0)
        x = x.t()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.softmax(self.l3(x))
        # from IPython import embed
        # embed()
        return x


def obj2id(model_name, color_id):
    if model_name == 'prim_cube':
        if color_id == 1:
            return -128.0
        elif color_id == 2:
            return -128.0
        elif color_id == 7:
            return -128.0
    if model_name == 'prim_sphere':
        if color_id == 3:
            return 48.0  # agent
        elif color_id == 6:
            return 128.0  # target
    print("ERROR in obj2id @", model_name, color_id)
    # from IPython import embed
    # embed()
    return 0.0


class Model:
    """ simple feed forward model """

    def __init__(self, game):
        # self.output_noise = game.output_noise
        self.policy_model = Policy(game.max_object, game.layers[0], game.layers[1]).double()
        self.max_objects = game.max_object
        self.env = None
        self.env_config = None

        # self.time_input = 0  # use extra sinusoid input
        # self.sigma_bias = game.noise_bias  # bias in stdev of output
        # self.sigma_factor = 0.5  # multiplicative in stdev of output
        # if game.time_factor > 0:
        #     self.time_factor = float(game.time_factor)
        #     self.time_input = 1
        # self.input_size = game.input_size
        # self.output_size = game.output_size
        # self.shapes = [(self.input_size + self.time_input, self.layer_1),
        #                (self.layer_1, self.layer_2),
        #                (self.layer_2, self.output_size)]

        # self.sample_output = False
        # if game.activation == 'relu':
        #     self.activations = [relu, relu, passthru]
        # elif game.activation == 'sigmoid':
        #     self.activations = [np.tanh, np.tanh, sigmoid]
        # elif game.activation == 'softmax':
        #     self.activations = [np.tanh, np.tanh, softmax]
        #     self.sample_output = True
        # elif game.activation == 'passthru':
        #     self.activations = [np.tanh, np.tanh, passthru]
        # else:
        #     self.activations = [np.tanh, np.tanh, np.tanh]

        # self.weight = []
        # self.bias = []
        # self.bias_log_std = []
        # self.bias_std = []
        # self.param_count = 0

        # idx = 0
        # for shape in self.shapes:
        #     self.weight.append(np.zeros(shape=shape))
        #     self.bias.append(np.zeros(shape=shape[1]))
        #     self.param_count += (np.product(shape) + shape[1])
        #     if self.output_noise[idx]:
        #         self.param_count += shape[1]
        #     log_std = np.zeros(shape=shape[1])
        #     self.bias_log_std.append(log_std)
        #     out_std = np.exp(self.sigma_factor * log_std + self.sigma_bias)
        #     self.bias_std.append(out_std)
        #     idx += 1

    def make_env(self, seed, env_config):
        # print("##### MODEL -> the env is the same?? ", env_config, "<>", self.env_config, " : ",
        #       env_config == self.env_config)
        if self.env_config == env_config:
            return
        if self.env:
            self.env.close()
        self.env_config = env_config
        # logger.info("making new env: ", env_config)
        print("#### MODEL -> making new env", env_config)
        self.env = gym.make('gym_tdw:tdw_puzzle_proc-v0', tdw_ip='169.48.98.28', self_ip="212.71.245.133")
        self.env.set_observation(True)
        # is_main_sc, no_target, no_cube_stack_target, no_cones_target,
        # no_walled_target, no_cubes, no_rectangles, no_cones, is_ramp_inside
        puzzle_data = create_puzzle_poet_TASK1(False,  # TODO: is main sc, ENABLE it later
                                               env_config.no_target,
                                               env_config.no_cube_stack_target,
                                               env_config.no_cones_target,
                                               env_config.no_walled_target,
                                               env_config.no_cube,
                                               env_config.no_rectangles,
                                               env_config.is_ramp_inside)
        display_table(puzzle_data)
        self.env.add_change_puzzle(puzzle_data)

    def get_action(self, obs, t=0, mean_mode=False):
        """
        :param obs:
        :param t:
        :param mean_mode:
        :return: get action from observation
        """

        obj_info = obs['object_information']
        obj_details = []

        print("---------")
        for value in obj_info.values():
            obj_detail = []
            x, y, z = value['position']
            obj_id = obj2id(value['model_name'], value['color_id'])
            obj_detail.append(x*128)
            obj_detail.append(y*128)
            obj_detail.append(z*128)
            obj_detail.append(obj_id)
            print(x*128, y*128, z*128, obj_id)
            obj_details.append(obj_detail)
        print("-----------")
        obj_len = len(obj_details)
        for _ in range(obj_len, self.max_objects):
            obj_details.append([0.0, 0.0, 0.0, 0.0])  # padding

        npx = np.asarray(obj_details)
        # print("## MODEL -> strange bug? ", npx)

        # npx = npx.transpose()
        tx = torch.from_numpy(npx).double()
        tx = tx.unsqueeze(0).unsqueeze(0).double()  # 1 * 4 * max_obj
        prob = self.policy_model(tx)

        print("MODEL -> simulating sample from ", prob.data.cpu().numpy()[0], end=' ')
        # logger.debug("MODEL -> simulate : ", prob)

        # print("## MODEL -> simulate : ", prob)
        action = prob.multinomial(num_samples=1).data.cpu().numpy()
        action = action[0][0]
        multipier = 20.0
        x_action = (action // 3 - 1) * multipier
        z_action = (action % 3 - 1) * multipier
        action = {'x': x_action, 'z': z_action}
        return action

    def set_model_params(self, model_params):
        self.policy_model.load_state_dict(model_params)
        # print( "####### MODEL -> set model param : ", model_params )
        # pointer = 0
        # for i in range(len(self.shapes)):
        #     w_shape = self.shapes[i]
        #     b_shape = self.shapes[i][1]
        #     s_w = np.product(w_shape)
        #     s = s_w + b_shape
        #     chunk = np.array(model_params[pointer:pointer + s])
        #     self.weight[i] = chunk[:s_w].reshape(w_shape)
        #     self.bias[i] = chunk[s_w:].reshape(b_shape)
        #     pointer += s
        #     if self.output_noise[i]:
        #         s = b_shape
        #         self.bias_log_std[i] = np.array(
        #             model_params[pointer:pointer + s])
        #         self.bias_std[i] = np.exp(
        #             self.sigma_factor * self.bias_log_std[i] + self.sigma_bias)
        #         pointer += s

    def load_model(self, filename):
        with open(filename) as f:
            model_params = torch.load(filename)
        # print('loading file %s' % (filename))
        self.set_model_params(model_params)

    def get_random_model_params(self):
        return self.policy_model.state_dict()


final_mode = False
RENDER_DELAY = False
record_video = False
MEAN_MODE = False
EMPTY_WINDOWS = 3


def get_reward(obs):
    """
    compute reward according to obs, - d( agent, the nearest goal )
    :param obs:
    :return: reward
    """
    obj_info = obs['object_information']

    agent_x = 0.0
    agent_y = 0.0
    agent_z = 0.0

    min_dist = 100000 # large enough

    for value in obj_info.values():
        x, y, z = value['position']
        if value['model_name'] == 'prim_sphere' and value['color_id'] == 3:
            agent_x = x
            agent_y = y
            agent_z = z
            break
    for value in obj_info.values():
        x, y, z = value['position']
        if (value['model_name'] == 'prim_sphere') and (value['color_id'] == 6):
            dist = (x - agent_x) ** 2 + (y - agent_y) ** 2 + (z - agent_z) ** 2
            # print("(x,y,z): ", x, y, z, " -> ", dist)
            if dist < min_dist:
                min_dist = dist
    # print( "## get reward : ", min_dist )
    # from IPython import embed
    # embed()
    reward = -min_dist
    if reward < -5:  # lower bound
        reward = -5
    return reward


# TODO maybe we can let returns to be the percent of goal reached
def simulate(model, seed, train_mode=False, num_episode=10,
             max_len=30, env_config_this_sim=None):
    """
    :param model:
    :param seed:
    :param train_mode:
    :param num_episode: # training episode
    :param max_len: in a single episode, the max length of agent action
    :param env_config_this_sim:
    :return:
    """
    reward_list = []
    t_list = []

    max_episode_length = 3  # TODO change here

    if train_mode and max_len > 0:
        if max_len < max_episode_length:
            max_episode_length = max_len

    if (seed >= 0):
        logger.debug('Setting seed to {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        # model.env.seed(seed)

    # print("##### MODEL -> simulate ")
    if env_config_this_sim:
        model.make_env(None, env_config_this_sim)

    for iter_num in range(num_episode):

        obs = model.env.reset()
        if obs is None:
            obs, _, _, _ = model.env.step({'x': 0, 'z': 0})

        total_reward = 0.0
        fake_done = False
        iter_len = 0
        for t in range(max_episode_length):
            action = model.get_action(obs, t=t, mean_mode=False)

            print("get action @ ", action)
            # print("####### MODEL -> simulate : ", action)

            obs, reward, done, info = model.env.step(action)
            reward *= 20
            for _ in range(EMPTY_WINDOWS):
                obs, reward_empty, done, info = model.env.step({'x': 0, 'z': 0})  # observation, reward, done, info
                reward += reward_empty * 20
                if done:  # if done == True or we have a reward
                    break
            if done and reward <= 0:  # extra punishment for jumping out of boundary
                reward = reward - 100
                fake_done = True
            reward += get_reward(obs)
            # img = obs['image']
            # img = Image.fromarray(img.astype(np.uint8))
            # img.save('./tdw_logs/views/'+env_config_this_sim.name+'_view_'+str(iter_num)+'_'+str(t)+'.png')
            total_reward += reward
            # print("####### MODEL -> reward: ", reward)

            if done:
                if fake_done:
                    logger.info("simulate @ fake done")
                    # print("####### MODEL -> simulate @ fake done")
                else:
                    logger.info("simulate @ done")
                    # print("####### MODEL -> simulate @ done")
                break
            iter_len += 1
        print("# MODEL -> total reward: ", total_reward)
        # from IPython import embed
        # embed()
        reward_list.append(total_reward)
        t_list.append(iter_len)

    return reward_list, t_list


if __name__ == "__main__":
    policy_nn = Policy(8, 16, 16).double()
    obj_details = []

    for value in range(3):
        obj_detail = [1.0 + value, 2.0 + value, 3.0 + value, 4.0 + value]
        obj_details.append(obj_detail)
    for _ in range(3,8):
        obj_detail = [0.0, 0.0, 0.0, 0.0]
        obj_details.append(obj_detail)

    npx = np.asarray(obj_details)
    # npx = npx.transpose()
    tx = torch.from_numpy(npx).double()
    tx = tx.unsqueeze(0).unsqueeze(0).double()  # 1 * 4 * max_obj

    y = policy_nn(tx)

    print(y)
    ym = y.multinomial(num_samples=1).data.cpu().numpy()
    print(ym)