from gym_tdw.envs.utils.aux_utils import load_scene_example, step_one_frame
from gym_tdw.envs.utils import primitives, control_tasks, procedurally_gen_control_tasks
from gym_tdw.envs.utils.object_utils import create_object
from os.path import join
from json import loads
import base64
from PIL import Image
import io
import numpy as np
from tdw.tdw_utils import TDWUtils
import requests
import pandas as pd
import os
import json
from tdw.controller import Controller
from gym_tdw.envs.utils.cos_ops import port_tracker
from gym_tdw.envs.utils.proc_gen import create_puzzle


class TDW_sim(Controller):
    def __init__(self, puzzle_number, port):
        self.objects = None
        self.puzzle_number = puzzle_number
        super().__init__(port=port)

    def run(self, puzzle_type="non-goal", proc_gen=False, debug=True):
        self.start()
        load_scene(self, debug)

    def take_action(self, action):
        return take_action(self, self.objects, action)

    def output_images(self, output):
        if output:
            self.communicate({"$type": "send_images",
                                 "frequency": "always"})
        else:
            self.communicate({"$type": "send_images",
                              "frequency": "never"})

    def change_material(self, object_id, material_id=None):
        if material_id:
            self.communicate(
                [{"$type": "set_visual_material", "id": object_id, "new_material_name": material_id,
                 "object_name": "PrimSphere_0",
                 "old_material_index": 0}, {"$type": "set_physic_material", "id": object_id, "dynamic_friction": 1,
                 "static_friction": 1,
                 "bounciness": 0.1}])

            return
        self.communicate(
            {"$type": "set_visual_material", "id": object_id, "new_material_name": "plastic_hammered",
             "object_name": "PrimSphere_0",
             "old_material_index": 0})

class SceneState:
    def __init__(self):
        self.collision_data = None
        self.collided = False
        self.object_data = None
        self.object_updated = False
        self.image_1 = None
        self.image_2 = None
        self.image_1_ready = False
        self.image_2_ready = False

    def set_collision_data(self, data):
        self.collision_data = data
        self.collided = True

    def set_object_data(self, data):
        self.object_data = data
        self.object_updated = True

    def parse_collision_data(self):
        collider_id = self.collision_data["collider_id"]
        collidee_id = self.collision_data["collidee_id"]
        return collider_id, collidee_id

    def parse_object_data(self):
        return_data = {}
        for objs in self.object_data:
            if "model_name" in objs.keys():
                return_data[objs["id"]] = {
                    "model_name": objs["model_name"],
                    "position": objs["position"],
                    "velocity": objs["velocity"],
                    "rotation": objs["rotation"],
                    "mass": objs["mass"]
                }
        return return_data

scene_state_data = SceneState()


class puzzle_state():
    def __init__(self, objects):
        self.target_sphere = {}

        self.init_tgt_sphere_reward_state(objects=objects)

        self.sphere = objects["sphere"]
        if "goal_boundaries" in objects:
            self.goal_boundaries = objects["goal_boundaries"]
        else:
            self.goal_boundaries = None

    def process_collision(self, collider_id, collidee_id):
        if collider_id == self.sphere and collidee_id in list(self.target_sphere.keys()):
            if self.target_sphere[collidee_id] != 1:
                self.target_sphere[collidee_id] = 1
                self.change_material(collidee_id)

    def init_tgt_sphere_reward_state(self, objects):
        for sphere in objects["target_spheres"]:
            self.target_sphere[sphere] = 0

    def get_reward(self, _object_data=None):
        if self.goal_boundaries:
            for tgt_sphere in self.target_sphere.keys():
                pos = _object_data[tgt_sphere]["position"]
                if self.goal_boundaries["x_left"] < pos["x"] < self.goal_boundaries["x_right"] and self.goal_boundaries["z_bottom"] < pos["z"] < self.goal_boundaries["z_top"]:
                    if self.target_sphere[tgt_sphere] != 1:
                        self.target_sphere[tgt_sphere] = 1
                        self.change_material(tgt_sphere)
        return sum(self.target_sphere.values())

    def change_material(self, object_id):
        tdw.send_to_server(
            {"$type": "set_visual_material", "id": object_id, "new_material_name": "plastic_hammered",
             "object_name": "PrimSphere_0",
             "old_material_index": 0})

    def episode_complete(self, _object_data=None):
        if _object_data is not None:

            for _sphere in list(self.target_sphere.keys()) + [self.sphere]:

                pos = _object_data[_sphere]["position"]

                if not(-4.718 < pos["x"] < -3.532 and -5.687 < pos["z"] < -3.527):
                    return True

        for key, value in self.target_sphere.items():
            if value != 1:
                return False
        return True


def load_scene(tdw_object, debug=True):
    if debug:
        tdw_object.communicate(TDWUtils.create_empty_room(24, 24))
    else:
        tdw_object.communicate(TDWUtils.create_empty_room(24, 24))
        # Load streamed scene.
        # tdw_object.load_streamed_scene(scene="tdw_room_2018")
        # Change the skybox
        tdw_object.communicate({"$type": "set_hdri_skybox", "skybox_name": "furry_clouds_4k"})
        # Adjust post-processing settings.
        tdw_object.communicate([
            {"$type": "set_post_exposure", "post_exposure": 1.25},
            {"$type": "set_contrast", "contrast": 0},
            {"$type": "set_saturation", "saturation": 10},
            {"$type": "set_screen_space_reflections", "enabled": True},
            {"$type": "set_vignette", "enabled": False}])
        # Set the shadow strength to maximum.
        tdw_object.communicate({"$type": "set_shadow_strength", "strength": 1.0})
    # set screen size
    tdw_object.communicate({"$type": "set_screen_size", "height": 480, "width": 640})


def load_initial_objects(tdw_object):
    tdw_object.communicate({"$type": "create_avatar",
                            "type": "A_Img_Caps_Kinematic",
                            "id": "uniqueid1"})
    # tdw_object.communicate({"$type": "teleport_avatar_to", "avatar_id": "uniqueid1", "env_id": 0,
    #                         "position": {"x": -4.858, "y": 2.169, "z": -6.496}})
    tdw_object.communicate({"$type": "teleport_avatar_to", "avatar_id": "uniqueid1", "env_id": 0,
                            "position": {"x": -0.7333, "y": 2.169, "z": -1.888}})
    tdw_object.communicate({"$type": "set_pass_masks", "avatar_id": "uniqueid1", "pass_masks": ["_img"]})
    tdw_object.communicate({"$type": "rotate_avatar_to_euler_angles", "avatar_id": "uniqueid1", "env_id": 0,
                            "euler_angles": {"x": 41.021, "y": 24.494, "z": 0}})

    # ramp = create_object(tdw_object, "billiardtable", {"x": -4.125461, "y": 0.0, "z": -4.608},
    #                      {"x": 0.0, "y": 0.0, "z": 0.0})
    ramp = create_object(tdw_object, "billiardtable", {"x": 0, "y": 0.0, "z": 0},
                         {"x": 0.0, "y": 0.0, "z": 0.0})
    tdw_object.communicate({"$type": "set_kinematic_state", "id": ramp, "is_kinematic": True, "use_gravity": False})
    # tdw_object.communicate([{"$type": "set_mass", "id": ramp, "mass": 10000000}, {"$type": "set_physic_material", "dynamic_friction": 0.6,
    #                         "static_friction": 0.6,
    #                         "bounciness": 1.5, "id": ramp}])
    table_center = (0, 0)
    primitives.thin_wall(tdw_object,  {"x": table_center[0] - 0.589, "y": 0.8749, "z": table_center[1] - 1.0725}, {"x": 0.0, "y": -45.0, "z": 0.0})
    primitives.thin_wall(tdw_object, {"x": table_center[0] + 0.589, "y": 0.8749, "z": table_center[1] - 1.0725}, {"x": 0.0, "y": 45.0, "z": 0.0})
    primitives.thin_wall(tdw_object, {"x": table_center[0] + 0.589, "y": 0.8749, "z": table_center[1] + 1.0725}, {"x": 0.0, "y": -45.0, "z": 0.0})
    primitives.thin_wall(tdw_object, {"x": table_center[0] - 0.589, "y": 0.8749, "z": table_center[1] + 1.0725}, {"x": 0.0, "y": 45.0, "z": 0.0})

    # Create mid pocket cones
    primitives.thin_wall(tdw_object, {"x": table_center[0] - 0.593, "y": 0.8749, "z": table_center[1] + 0.0454}, {"x": 0.0, "y": -45.0, "z": 0.0})
    primitives.thin_wall(tdw_object, {"x": table_center[0] - 0.593, "y": 0.8749, "z": table_center[1] - 0.0454}, {"x": 0.0, "y": 45.0, "z": 0.0})

    primitives.thin_wall(tdw_object, {"x": table_center[0] + 0.593, "y": 0.8749, "z": table_center[1] + 0.0454}, {"x": 0.0, "y": 45.0, "z": 0.0})
    primitives.thin_wall(tdw_object, {"x": table_center[0] + 0.593, "y": 0.8749, "z": table_center[1] - 0.0454}, {"x": 0.0, "y": -45.0, "z": 0.0})


def setup_tdw_instance(tdw_ip, self_ip):
    # url = "http://52.116.149.125:5000/get_tdw"
    url = "http://{}:5000/get_tdw".format(tdw_ip)
    available_port = get_port()
    data = {
        'ip_address': self_ip,
        'port': int(available_port)
    }
    response = requests.post(url, json=json.dumps(data))
    print(response.status_code, response.reason)
    docker_id = response.json()['docker_id']
    return docker_id, available_port


def get_port():
    if os.path.isfile("available_ports.csv"):
        available_ports = pd.read_csv("available_ports.csv")
    else:
        print("Port tracking file doesn't exist. Creating one..")
        available_ports = pd.DataFrame(columns=["port", "status"])
        no_ports = 100
        port_start = 1071
        for i in range(no_ports):
            available_ports.loc[i] = [port_start, "free"]
            port_start += 1
    available_port_ = None
    for i in range(available_ports.shape[0]):
        if available_ports['status'].iloc[i] == "free":
            available_port_ = available_ports["port"].iloc[i]
            available_ports["status"].iloc[i] = "not-free"
            break
    if not available_port_:
        raise Exception("No port available")
    available_ports.to_csv("available_ports.csv", index=False)
    return available_port_


def kill_tdw(tdw_ip, docker_id):
    url = "http://{}:5000/kill_tdw".format(tdw_ip)
    data = {
        "container_id": docker_id
    }
    response = requests.post(url, json=json.dumps(data))
    print(response.status_code, response.reason, )


def free_port(port):
    available_ports = pd.read_csv("available_ports.csv")
    for i in range(available_ports.shape[0]):
        if available_ports['port'].iloc[i] == port:
            available_ports["status"].iloc[i] = "free"
            break
    available_ports.to_csv("available_ports.csv", index=False)
    return




def load_puzzle_proc_gen(tdw_object, puzzle_data):
    objects, puzzle_type = procedurally_gen_control_tasks.render_puzzle(tdw_object, puzzle_data)
    return objects, puzzle_type


def load_puzzle(tdw_object, puzzle_number):
    if puzzle_number == 1:
        # objects = control_tasks.puzzle_1(tdw_object)
        objects = control_tasks.puzzle_1(tdw_object)
        return objects, "non-goal"
    elif puzzle_number == 2:
        objects = control_tasks.puzzle_2(tdw_object)
        return objects, "non-goal"
    elif puzzle_number == 3:
        objects = control_tasks.puzzle_3(tdw_object)
        return objects, "non-goal"
    elif puzzle_number == 4:
        objects = control_tasks.puzzle_4(tdw_object)
        return objects, "goal"
    elif puzzle_number == 5:
        objects = control_tasks.puzzle_5(tdw_object)
        return objects, "non-goal"
    elif puzzle_number == 6:
        objects = control_tasks.puzzle_6(tdw_object)
        return objects, "goal"
    elif puzzle_number == 7:
        objects = control_tasks.puzzle_7(tdw_object)
        return objects, "non-goal"


def take_action(tdw_object, objects, actions):
    # If there is no force skip sending this command to save computation
    if "stop" in actions:
        if actions["stop"]:
            resp = tdw_object.communicate({"$type": "stop_object", "id": objects["sphere"]})
            return resp
    if actions["x"] != 0 or actions["z"] != 0:
        resp = tdw_object.communicate({"$type": "apply_force_to_object", "force": {"x": actions["x"], "y": 0, "z": actions["z"]},
                            "id": objects["sphere"]})
    else:
        resp = tdw_object.communicate({"$type": "step_physics", "frames": 0})
    return resp


def reset_scene(tdw_object, objects):
    reset_params = objects["reset_params"]
    commands = []
    for object_id in reset_params.keys():
        # Check if object is a lever and reset the position and orientation
        if "position" in reset_params[object_id].keys() and "rotation" in reset_params[object_id].keys():
            commands.extend([{"$type": "stop_object", "id": object_id},
                             {"$type": "rotate_object_to_euler_angles", "euler_angles": reset_params[object_id]["rotation"],
                              "id": object_id},
                             {"$type": "teleport_object", "id": object_id,
                              "position": reset_params[object_id]["position"]}
                             ])
        # Check if object is target sphere then reset and color and reset position and
        # default the orientation to {"x": 0, "y": 0, "z": 0}
        elif object_id in objects["target_spheres"]:
            commands.extend([{"$type": "stop_object", "id": object_id},
                                {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": 0, "z": 0},
                                 "id": object_id},
                                {"$type": "teleport_object", "id": object_id,
                                 "position": reset_params[object_id]},
                                {"$type": "set_visual_material", "id": object_id,
                                 "new_material_name": "plastic_vinyl_glossy_yellow",
                                 "object_name": "PrimSphere_0",
                                 "old_material_index": 0}
                                ])
        elif object_id in objects["push_spheres"]:
            commands.extend([{"$type": "stop_object", "id": object_id},
                             {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": 0, "z": 0},
                              "id": object_id},
                             {"$type": "teleport_object", "id": object_id,
                              "position": reset_params[object_id]},
                             {"$type": "set_visual_material", "id": object_id,
                              "new_material_name": "car_iridescent_paint",
                              "object_name": "PrimSphere_0",
                              "old_material_index": 0}
                             ])
        # For everything else just reset position and default the orientation to {"x": 0, "y": 0, "z": 0}
        else:
            commands.extend([{"$type": "stop_object", "id": object_id},
                                {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": 0, "z": 0},
                                 "id": object_id},
                                {"$type": "teleport_object", "id": object_id,
                                 "position": reset_params[object_id]}
                                ])
        tdw_object.communicate(commands)
