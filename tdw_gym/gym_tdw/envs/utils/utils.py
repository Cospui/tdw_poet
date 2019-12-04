import itertools
import utils.avatar_utils as avatar_utils
from gym_tdw.envs.utils.aux_utils import load_scene_example, on_recv, on_send, step_one_frame, change_output
import utils.object_utils as object_utils
import numpy as np
import TDW.tdw as tdw
import quaternion
import math


def generate_avatar_and_object_list():
    output = []
    avatar_list = ["Position_1", "Position_2", "Position_3"]
    object_list = ["obj_1", "obj_2", "obj_3", "obj_4"]
    three_object_combinations = generate_combinations(object_list, 3)
    four_object_combinations = generate_combinations(object_list, 4)
    for element in itertools.product(avatar_list, three_object_combinations):
        output.append(element)
    for element in itertools.product(avatar_list, four_object_combinations):
        output.append(element)
    return output


def generate_combinations(a, k):
    final_output = []
    if k == 1:
        return [[x] for x in a]
    else:
        for y,j in enumerate(a[:-(k-1)]):
            perms = generate_combinations(a[y+1:], k-1)
            output = [[j] + x for x in perms]
            final_output.extend(output)
    return final_output


def execute_commands(command_list, experiment_parameters):
    for cmd in command_list:
        if "step_frame" in cmd[0]:
            step_one_frame(cmd[1])
        elif cmd[0] == "apply_force_to_object":
            object_utils.apply_force_object(experiment_parameters["selected_object"], experiment_parameters["force"]*np.array(experiment_parameters["direction"]))
        else:
            avatar_utils.bend_arm(cmd[0], cmd[2], cmd[1], experiment_parameters["mitten"])
    tdw.send_to_server({"$type": "get_objects_data"})


def execute_commands2(command_list, experiment_parameters):
    frames = 0
    for cmd in command_list:
        if "step_frame" in cmd[0]:
            step_one_frame(cmd[1])
            frames += cmd[1]
        elif cmd[0] == "apply_force_to_object":
            object_utils.apply_force_object(experiment_parameters["selected_object"], experiment_parameters["force"]*np.array(experiment_parameters["direction"]))
            frames += 1
        else:
            avatar_utils.bend_arm(cmd[0], cmd[2], cmd[1], experiment_parameters["mitten"])
            frames += 1
    return frames

def generate_static_parameters(physical_parameters):
    output = [e for e in itertools.product(physical_parameters["mass"], physical_parameters["bounciness"], physical_parameters["friction"])]
    output = [e for e in itertools.product(output, output, output)]
    return output


def generate_dynamic_parameters(physical_parameters):
    directions = [(physical_parameters["directions"][k], "hand_b") if int(k) < 5 else (physical_parameters["directions"][k], "hand_f") for k in physical_parameters["directions"].keys()]
    output = [e for e in itertools.product(physical_parameters["force"], directions )]
    return output


def to_euler_angle(q):
    quat = np.quaternion(q["w"], q["x"], q["y"], q["z"])
    euler_angles = quaternion.as_euler_angles(quat)
    rotation_matrix = quaternion.as_rotation_matrix(quat)

    y = math.asin(rotation_matrix[0,2])
    if math.fabs(rotation_matrix[0,2]) < 0.99999:
        x = math.atan2(-1* rotation_matrix[1,2], rotation_matrix[2,2])
        z = math.atan2(-1 * rotation_matrix[0,1], rotation_matrix[0,0])
    else:
        x = math.atan2(rotation_matrix[2,1], rotation_matrix[1,1])
        z = 0

    return {"x":math.degrees(x), "y": math.degrees(y), "z":math.degrees(z)}