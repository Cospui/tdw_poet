from os.path import isdir, join
from os import mkdir, getcwd
import os
import utils.avatar_utils as avatar_utils
import utils.object_utils as object_utils
import utils.utils as utils
import random


room_created = False
table_created = False
objects_exist = False


def _create_room(room_length, room_width, room_height):
    """
    Example of how to load a scene with avatars.
    """

    # Load the scene.
    tdw.send_to_server([{"$type": "load_scene", "scene_name": "ProcGenScene"},  {"$type": "set_pause", "value": True}])

    # Build the rooms.
    assert type(room_length) == int, "Please enter integer"
    assert type(room_width) == int, "Please enter integer"
    assert type(room_height) == int, "Please enter integer"

    tdw.send_to_server({"$type": "build_rooms", "num_rooms_x": 1, "num_rooms_z": 1, "room_height": room_height,
                        "room_length": room_length, "room_width": room_width, "windows": True})
    tdw.send_to_server({"$type": "set_screen_size", "height": 960, "width": 1280})


def create_experiment_setup(experiment_parameters):
    _create_experiment_setup(experiment_parameters)


def _create_experiment_setup(experiment_parameters):
    """
    1. Given P, obj_1...obj_n, & physical properties
    2. Render Objects
    :param experiment_parameters:
    :return:
    """
    # TODO: Work on applying physical material - Done
    experiment_parameters = object_utils.create_objects(experiment_parameters)
    global objects_exist
    objects_exist = False
    return experiment_parameters


def create_initial_setup():
    """
    This method sets initial scene objects that remain constant for the entirety of this script
    1. Create the room
    3. Create Camera
    :param experiment_parameters:
    :return:
    """
    _create_room(8, 10, 1)
    tdw.send_to_server({"$type": "init_enviro_sky"})
    tdw.send_to_server({"$type": "set_time_of_day", "hour": 13.0})
    tdw.send_to_server({"$type": "set_render_quality", "render_quality": 1})
    experiment_parameters = {}
    # Create 3-D camera
    camera = "camera_1"
    avatar_utils.create_avatar("A_Img_Caps_Kinematic", {"x": -4.858, "y": 2.169, "z": -6.496}, camera)
    avatar_utils.set_pass_mark(camera)
    tdw.send_to_server({"$type": "rotate_avatar_to_euler_angles", "avatar_id": camera, "env_id": 0,
                        "euler_angles": {"x": 41.021, "y": 24.494, "z": 0}})
    experiment_parameters["camera1"] = camera

    # Create top camera
    camera = "camera_2"
    avatar_utils.create_avatar("A_Img_Caps_Kinematic", {"x": -4.194, "y": 4.108, "z": -4.626}, camera)
    avatar_utils.set_pass_mark(camera)
    tdw.send_to_server({"$type": "rotate_avatar_to_euler_angles", "avatar_id": camera, "env_id": 0,
                        "euler_angles": {"x": 90, "y": 0, "z": 0}})
    experiment_parameters["camera2"] = camera

    table = object_utils.create_object("billiardtable", {"x": -4.125461, "y": 0.0, "z": -4.608},
                                       {"x": 0.0, "y": 0.0, "z": 0.0})
    tdw.send_to_server({"$type": "set_mass", "id": table,
                        "mass": 10000.0})
    global table_created
    table_created = True
    experiment_parameters["table"] = table
    global room_created
    room_created = True
    return experiment_parameters


def reset_object_positions(experiment_parameters):
    object_utils.teleport_objects(experiment_parameters)


def reset_object_rotations(experiment_parameters):
    pass


def reset_avatar_rotation(experiment_parmeters):
    # avatar_utils.rotate_avatar(experiment_parmeters["mitten"], experiment_parmeters["avatar"]["avatar_rotation"])
    tdw.send_to_server({"$type": "rotate_avatar_to_euler_angles", "avatar_id": experiment_parmeters["mitten"], "env_id":0, "euler_angles": {"x": 0, "y": -90, "z": 0}}
)


def get_object_data():
    tdw.send_to_server({"$type": "get_objects_data"})