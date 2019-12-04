from gym_tdw.envs.tdw_env import TdwEnv
import os
from gym_tdw.envs.utils import gym_utils

class TdwEnv_puzzle_1_proc(TdwEnv):

    def __init__(self, tdw_ip, self_ip, debug=False):
        print("Creating new Environment")
        # self.pro_gen_puzzle_no = pro_gen_puzzle_no
        self.tdw_ip = tdw_ip
        self.self_ip = self_ip
        if tdw_ip == "localhost":
            self.port = "1071"
        else:
            self.tdw_docker_id, self.port = gym_utils.setup_tdw_instance(self.tdw_ip, self.self_ip)
        print(f"Connecting with tdw on port {self.port}")
        self.tdw_instance = gym_utils.TDW_sim(None, self.port)
        self.puzzle_type = None
        self.tdw_instance.run(proc_gen=True, debug=debug)
        self.output_images = False
        self.reward_tracker = {}
        self.puzzle_loaded = False
        self.episode = False

    def add_change_puzzle(self, puzzle_data, debug=False):
        if self.puzzle_loaded:
            self.tdw_instance.communicate({"$type": "destroy_all_objects"})
            if "highlighted_areas" in self.tdw_instance.objects.keys():
                for painting_id in list(self.tdw_instance.objects["highlighted_areas"].keys()):
                    self.tdw_instance.communicate({"$type": "destroy_painting", "id": painting_id})

        gym_utils.load_initial_objects(self.tdw_instance)
        self.tdw_instance.objects, self.puzzle_type = gym_utils.load_puzzle_proc_gen(self.tdw_instance, puzzle_data)
        self.init_reward(self.tdw_instance.objects)
        self.puzzle_loaded = True
        objects = list(self.tdw_instance.objects["model_names"].keys())

        self.tdw_instance.communicate({"$type": "send_collisions", "enter": True, "exit": True, "stay": True})
        self.tdw_instance.communicate([{"$type": "send_transforms", "frequency": "always", "ids": objects},
                          {"$type": "send_rigidbodies", "frequency": "always", "ids": objects}
                          ])
        self.episode = False