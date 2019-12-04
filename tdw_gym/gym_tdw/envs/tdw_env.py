import gym
from gym_tdw.envs.utils import gym_utils, object_configuration
import random
import time
import os

from tdw.output_data import Collision
from tdw.tdw_utils import TDWUtils
from tdw.output_data import Images, OutputData, Collision, Transforms, Rigidbodies, Bounds
from PIL import Image as pil_Image
import io
import numpy as np


class TdwEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, tdw_ip, self_ip, debug=True):
        print("Creating new Environment")
        self.tdw_ip = tdw_ip
        if tdw_ip == "localhost":
            self.port = "1071"
        else:
            self.tdw_docker_id, self.port = gym_utils.setup_tdw_instance(tdw_ip, self_ip)
        print(f"Connecting with tdw on port {self.port}")
        self.tdw_instance = gym_utils.TDW_sim(1, self.port)
        self.puzzle_type = "non-goal"
        self.tdw_instance.run(debug=debug)
        self.output_images = False
        self.reward_tracker = {}
        self.puzzle_loaded = False
        self.episode = False
        self.object_affected = []
        print("Tdw initialised")

    def step(self, action):
        """
            Parameters
            ----------
            action :

            Returns
            -------
            ob, reward, episode_over, info : tuple
                ob (object) :
                    an environment-specific object representing your observation of
                    the environment.
                reward (float) :
                    amount of reward achieved by the previous action. The scale
                    varies between environments, but the goal is always to increase
                    your total reward.
                episode_over (bool) :
                    whether it's time to reset the environment again. Most (but not
                    all) tasks are divided up into well-defined episodes, and done
                    being True indicates the episode has terminated. (For example,
                    perhaps the pole tipped too far, or you lost your last life.)
                info (dict) :
                     diagnostic information useful for debugging. It can sometimes
                     be useful for learning (for example, it might contain the raw
                     probabilities behind the environment's last state change).
                     However, official evaluations of your agent are not allowed to
                     use this for learning.
        """
        resp = self.tdw_instance.take_action(action)
        # images = Images(resp[0])
        obs = {}
        obs['object_information'] = {}
        for r in resp:
            r_id = OutputData.get_data_type_id(r)

            if r_id == "imag":
                images = Images(r)
                obs['image'] = np.array(pil_Image.open(io.BytesIO(images.get_image(0))))
            if r_id == "coll":
                self.process_collision(Collision(r))
            if r_id == "rigi":
                rigi_body_data = Rigidbodies(r)
                for object_index in range(rigi_body_data.get_num()):
                    if rigi_body_data.get_id(object_index) not in  obs['object_information'].keys():
                        obs['object_information'][rigi_body_data.get_id(object_index)] = {}
                    obs['object_information'][rigi_body_data.get_id(object_index)][
                        'velocity'] = rigi_body_data.get_velocity(object_index)
                    obs['object_information'][rigi_body_data.get_id(object_index)][
                        'mass'] = rigi_body_data.get_mass(object_index)
                    obs['object_information'][rigi_body_data.get_id(object_index)][
                        'angular_velocity'] = rigi_body_data.get_angular_velocity(object_index)
            if r_id == "tran":
                transform_data = Transforms(r)
                for object_index in range(transform_data.get_num()):
                    if transform_data.get_id(object_index) not in obs['object_information'].keys():
                        obs['object_information'][transform_data.get_id(object_index)] = {}
                    obs['object_information'][transform_data.get_id(object_index)][
                        'position'] = transform_data.get_position(object_index)
                    obs['object_information'][transform_data.get_id(object_index)][
                        'rotation'] = transform_data.get_rotation(object_index)

        for key in obs["object_information"].keys():
            obs["object_information"][key]["model_name"] = self.tdw_instance.objects["model_names"][key]
        if self.puzzle_type == 'goal' or self.puzzle_type == "hybrid":
            reward = self.get_reward(obs['object_information'])
        else:
            reward = self.get_reward()
        self.add_color_id(obs)
        self.update_episode_state(obs['object_information'])
        return obs, reward, self.episode, None

    def add_color_id(self, obs):
        object_configs = object_configuration.object_configuration()
        for key in obs["object_information"].keys():
            # if main sphere
            if key == self.tdw_instance.objects["sphere"]:
                obs["object_information"][key]["color_id"] = object_configs.main_sphere["color_id"]
            # If it's a push sphere
            elif key in self.tdw_instance.objects["push_spheres"]:
                if self.reward_tracker[key] != 0:
                    obs["object_information"][key]["color_id"] = object_configs.push_sphere["after_color_id"]
                else:
                    obs["object_information"][key]["color_id"] = object_configs.push_sphere["before_color_id"]
            # if it's push sphere
            elif key in self.tdw_instance.objects['target_spheres']:
                if self.reward_tracker[key] != 0:
                    obs["object_information"][key]["color_id"] = object_configs.touch_sphere["after_color_id"]
                else:
                    obs["object_information"][key]["color_id"] = object_configs.touch_sphere["before_color_id"]
            # If it's a cube
            elif obs["object_information"][key]["model_name"] == "prim_cube":
                if obs["object_information"][key]["mass"] == object_configs.cube_1["mass"]:
                    obs["object_information"][key]["color_id"] = object_configs.cube_1["color_id"]
                elif obs["object_information"][key]["mass"] == object_configs.cube_2["mass"]:
                    obs["object_information"][key]["color_id"] = object_configs.cube_2["color_id"]
            else:
                print(f"{key} not found")

    def stop_all_objects_inside_goal(self):

        for key, value in self.reward_tracker.items():
            if key in self.tdw_instance.objects["push_spheres"] and self.reward_tracker[key] == 1:
                self.tdw_instance.communicate({"$type": "stop_object", "id": key})

    def check_highlight(self, object_information):
        # Test: Check if the physical material changes fast enough
        x = -3.776
        z = -5.472
        diff = 0.304
        for tgt_sphere in self.tdw_instance.objects['target_spheres'] + [self.tdw_instance.objects['sphere']] + self.tdw_instance.objects["push_spheres"]:
            pos = object_information[tgt_sphere]["position"]
            if (x - diff) < pos[0] < (x + diff) and (z - diff) < pos[2] < (z + diff) and tgt_sphere not in self.object_affected:
                print("Object in highlighted area.... changing physical params")
                self.object_affected.append(tgt_sphere)
                self.tdw_instance.communicate({"$type": "set_physic_material", "id": tgt_sphere, "dynamic_friction": 1,
                                               "static_friction": 1, "bounciness": 0.5})

    def process_collision(self, collision_data):
        objects = self.tdw_instance.objects
        if "walls" in objects.keys():
            wall_ids = objects["walls"]
            if (collision_data.get_collider_id() in wall_ids or collision_data.get_collidee_id() in wall_ids) and (objects["sphere"] == collision_data.get_collider_id() or objects["sphere"] == collision_data.get_collidee_id()):
                self.episode = True
                self.reset()
        if (objects["sphere"] == collision_data.get_collider_id() and collision_data.get_collidee_id() in objects["target_spheres"]) or (objects["sphere"] == collision_data.get_collidee_id() and collision_data.get_collider_id() in objects["target_spheres"]):
            self.update_reward(collision_data.get_collidee_id() if collision_data.get_collidee_id() in objects["target_spheres"] else collision_data.get_collider_id())

    def update_episode_state(self, object_information=None):
        # If episode is already done than return
        if self.episode:
            return
        # Makesure all balls are on the table else end the episode
        for _sphere in self.tdw_instance.objects['target_spheres'] + [self.tdw_instance.objects['sphere']] + self.tdw_instance.objects['push_spheres']:
            pos = object_information[_sphere]["position"]
            if not (-0.706 < pos[0] < 0.706 and -1.187 < pos[2] < 1.187):
                self.episode = True
                return
        # Check reward tracker
        for key, value in self.reward_tracker.items():
            if value != 1:
                self.episode = False
                return
        self.episode = True

    def update_reward(self, tgt_id):
        if self.reward_tracker[tgt_id] == 0:
            self.reward_tracker[tgt_id] = 1
            self.tdw_instance.change_material(tgt_id)

    def add_change_puzzle(self, puzzle_number):
        if self.puzzle_loaded:
            self.tdw_instance.communicate({"$type": "destroy_all_objects"})

            if "highlighted_areas" in self.tdw_instance.objects.keys():
                for painting_id in list(self.tdw_instance.objects["highlighted_areas"].keys()):
                    self.tdw_instance.communicate({"$type": "destroy_painting", "id": painting_id})
        gym_utils.load_initial_objects(self.tdw_instance)
        self.tdw_instance.objects, self.puzzle_type = gym_utils.load_puzzle(self.tdw_instance, puzzle_number=puzzle_number)
        objects = list(self.tdw_instance.objects["model_names"].keys())


        self.tdw_instance.communicate({"$type": "send_collisions", "enter": True, "exit": True, "stay": True})
        self.tdw_instance.communicate([{"$type": "send_transforms", "frequency": "always", "ids": objects},
                          {"$type": "send_rigidbodies", "frequency": "always", "ids": objects}
                          ])
        if "highlighted_areas" in self.tdw_instance.objects.keys():
            self.tdw_instance.communicate({"$type": "send_bounds", "frequency": "always",
                              "ids": list(self.tdw_instance.objects["highlighted_areas"].keys())})
        self.init_reward(self.tdw_instance.objects)
        self.episode = False
        self.puzzle_loaded = True

    def get_reward(self, object_information=None):
        if 'goal_boundaries' in self.tdw_instance.objects:
            for tgt_sphere in self.tdw_instance.objects['push_spheres']:
                pos = object_information[tgt_sphere]["position"]
                goal_boundaries = self.tdw_instance.objects["goal_boundaries"]
                if goal_boundaries["x_left"] < pos[0] < goal_boundaries["x_right"] and goal_boundaries["z_bottom"] < \
                        pos[2] < goal_boundaries["z_top"]:
                    if self.reward_tracker[tgt_sphere] != 1:
                        self.reward_tracker[tgt_sphere] = 1
                        self.tdw_instance.change_material(tgt_sphere, "metallic_car_paint")
                # If the ball has reward but it is out of goal turn it back
                elif self.reward_tracker[tgt_sphere] == 1:
                    self.tdw_instance.change_material(tgt_sphere, "car_iridescent_paint")
                    self.reward_tracker[tgt_sphere] = 0
        return sum(self.reward_tracker.values())

    def sample_random_action(self):
        return {
            "x": random.randint(-50, 50),
            "z": random.randint(-50, 50)
        }

    def set_observation(self, output=False):
        self.tdw_instance.output_images(output)

    def init_reward(self, objects):
        self.reward_tracker = {}
        for sphere in objects["target_spheres"] + objects["push_spheres"]:
            self.reward_tracker[sphere] = 0

    def reset(self):
        gym_utils.reset_scene(self.tdw_instance, self.tdw_instance.objects)
        self.init_reward(self.tdw_instance.objects)
        self.episode = False

    def _render(self):
        pass

    def _get_reward(self):
        """ Reward is given for XY. """
        pass

    def _seed(self):
        pass

    def get_maximum_reward(self):
        return len(self.reward_tracker.keys())

    def close(self):
        if self.tdw_ip != "localhost":
            gym_utils.kill_tdw(self.tdw_ip, self.tdw_docker_id)
        self.tdw_instance = None
        gym_utils.free_port(self.port)