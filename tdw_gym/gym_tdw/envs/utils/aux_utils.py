

from os.path import isdir, join
import random
from os import mkdir, getcwd
from json import loads
import os
import json


exp_no = 0
directory = join(getcwd(), "dist")
if not isdir(directory):
    mkdir(directory)
obj_list = []
ready = False
image_count1 = 0
image_count2 = 0
objects = []
avatar = []
avatar_ball_collision = False
objects_ready = False
object_data = None
task = None



def load_scene_example(room_length, room_width, room_height, avatar="A_Img_Caps", avatar_position={"x": -0.9, "y": 0, "z": -2.0}, dir_name=None):
    """
    Example of how to load a scene with avatars.
    """

    # Load the scene.
    tdw.send_to_server([{"$type": "load_scene", "scene_name": "ProcGenScene"},  {"$type": "set_pause", "value": True}])
    # tdw.send_to_server({"$type": "set_render_quality", "render_quality": 1})
    # tdw.send_to_server({"$type": "init_enviro_sky"})
    # tdw.send_to_server({"$type": "set_post_process", "value": True})
    # tdw.send_to_server({"$type": "set_time_of_day", "hour": 13.0})
    # tdw.send_to_server({"$type": "set_proc_gen_floor_material", "material_name": "concrete_01"})
    # tdw.send_to_server({"$type": "set_proc_gen_floor_texture_scale", "scale": {"x": 2, "y": 4}})
    global directory, task
    task = dir_name
    if dir_name:
        directory = join(getcwd(), "dist", dir_name)
        if not os.path.isdir(directory):
            mkdir(directory)
    if not os.path.isdir(os.path.join(directory, "camera_1")):
        mkdir(os.path.join(directory, "camera_1"))
        mkdir(os.path.join(directory, "camera_1", "FollowCamera"))
        mkdir(os.path.join(directory, "camera_1", "SensorContainer"))
    if not os.path.isdir(os.path.join(directory, "camera_2")):
        mkdir(os.path.join(directory, "camera_2"))
        mkdir(os.path.join(directory, "camera_2", "FollowCamera"))
        mkdir(os.path.join(directory, "camera_2", "SensorContainer"))


    # Build the rooms.
    assert type(room_length) == int, "Please enter integer"
    assert type(room_width) == int, "Please enter integer"
    assert type(room_height) == int, "Please enter integer"

    tdw.send_to_server({"$type": "build_rooms", "num_rooms_x": 1, "num_rooms_z": 1, "room_height": room_height,
                        "room_length": room_length, "room_width": room_width, "windows": True})


    # tdw.send_to_server([{"$type": "create_avatar", "type": avatar, "id": "uniqueid0"},
    #                     {"$type": "teleport_avatar_to", "avatar_id": "uniqueid0", "env_id": 0,
    #                      "position": avatar_position},
    #                     {"$type": "set_avatar_output", "avatar_id": "uniqueid0", "env_id": 0, "images": False,
    #                      "object_info": True, "avatar_info": True, "child_info": True,
    #                      "collision_info": True, "sensors_info": True}
    #                     ])
    # Set the pass masks of the cameras. This must be sent separately from the avatar creation command.
    # tdw.send_to_server({"$type": "set_pass_masks", "avatar_id": "uniqueid0", "pass_masks": ["_img"]})


def urban_scene(avatar="A_Img_Caps", avatar_position={"x": -0.9, "y": 0, "z": -2.0}, dir_name=None):
    """
    Example of how to load a scene with avatars.
    """

    # Load the scene.
    tdw.send_to_server([{"$type": "load_streamed_scene", "scene_name": "suburb_scene"},  {"$type": "set_pause", "value": True}])
    global directory
    if dir_name:
        directory = join(getcwd(), "dist", dir_name)
        if not os.path.isdir(directory):
            mkdir(directory)
    if not os.path.isdir(os.path.join(directory, "FollowCamera")):
        mkdir(os.path.join(directory, "FollowCamera"))
    if not os.path.isdir(os.path.join(directory, "SensorContainer")):
        mkdir(os.path.join(directory, "SensorContainer"))



    tdw.send_to_server([{"$type": "create_avatar", "type": avatar, "id": "uniqueid0"},
                        {"$type": "teleport_avatar_to", "avatar_id": "uniqueid0", "env_id": 0,
                         "position": avatar_position},
                        {"$type": "set_avatar_output", "avatar_id": "uniqueid0", "env_id": 0, "images": False,
                         "object_info": True, "avatar_info": True, "child_info": True,
                         "collision_info": True, "sensors_info": True}
                        ])
    # Set the pass masks of the cameras. This must be sent separately from the avatar creation command.
    tdw.send_to_server({"$type": "set_pass_masks", "avatar_id": "uniqueid0", "pass_masks": ["_img"]})


def set_avatar_output(image=False, sensor=False):
    assert type(image) == bool, "The value has to be bool"
    assert type(sensor) == bool, "The value has to be bool"
    tdw.send_to_server({"$type": "set_avatar_output", "avatar_id": "uniqueid0", "env_id": 0, "images": image,
                        "object_info": True, "avatar_info": True, "child_info": True,
                        "collision_info": True, "sensors_info": sensor}, )


def step_one_frame(tdw_object, n):
    for i in range(n):
        tdw_object.communicate({"$type": "do_nothing"})


def look_at(obj_id):
    tdw.send_to_server({"$type": "look_at", "avatar_id": "uniqueid0", "env_id": 0, "object_id": obj_id})


def on_recv(messages):
    """
    Example on_recv callback.

    :param messages: The received messages.
    """

    # Create the image save directory.
    global avatar_ball_collision, object_data, objects_ready, image_count1,image_count2, task
    tdw.global_message_count += 1

    for message_dict in messages:
        # Convert the byte array to a JSON object.
        message_dict = loads(message_dict)

        if message_dict["$type"] == "objects_data":
            object_data = message_dict["objects"]
            objects_ready = True


        # Grab image data and print meta-data.
        if message_dict["$type"] == "image_data":
            # Print the metadata.

            # Print the passes and output the images.

            for image_pass in message_dict["passes"]:
                if message_dict["avatar_id"] == "uniqueid1":
                    camera_name = "camera_1"
                else:
                    camera_name = "camera_2"
                # Save the image.
                if message_dict["sensor_name"] in "SensorContainer":
                    path = join("dist", task, camera_name, "SensorContainer", "image-{0}".format(image_count1 if camera_name == "camera_1" else image_count2) +
                                image_pass["extension"])
                else:
                    path = join("dist", task, camera_name, "FollowCamera", "image-{0}".format(image_count1 if camera_name == "camera_1" else image_count2) +
                                image_pass["extension"])
                tdw.save_img(image_pass["image"], path)
                if camera_name == "camera_1":
                    image_count1 += 1
                else:
                    image_count2 += 1


def on_send():
    """
    Example on_send callback.
    """
    print("Sent a message!")
    # pass


def change_output(image=False):
    tdw.send_to_server({"$type": "set_avatar_output", "avatar_id": "uniqueid0", "env_id": 0, "images": image,
                         "object_info": True, "avatar_info": True, "child_info": True,
                         "collision_info": True, "sensors_info": True})


def apply_force_rand(obj_id_list):
    for obj_id in obj_id_list:
        tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": random.randint(-20,20), "y": random.randint(-5,5), "z": random.randint(-20,20)}, "id": obj_id})


def apply_force(obj_id, x, y, z):
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": x, "y": y, "z": z}, "id": obj_id})


def create_object_example(loc, rot):
    """
    Example of how to create an object with a semantic material.
    """

    # Note that the object_id must be the same in both cases.
    object_id = tdw.get_unique_id()
    # tdw.send_to_server({"$type": "add_random_objects", "env_id": 0, "num_objects": 1})
    tdw.send_to_server({"$type": "add_objects", "env_id": 0, "objects":
        [{"name": "alma_floor_lamp", "position": loc, "orientation":rot, "id": object_id}]})
    # tdw.send_to_server({"$type": "set_semantic_material_to_default", "id": object_id})
    return object_id


def destroy_all_objects():
    tdw.send_to_server({"$type": "destroy_all_objects"})


def destroy_object(obj_id_list):
    for obj_id in obj_id_list:
        tdw.send_to_server({"$type": "destroy_object", "id": obj_id})


def update_object_info():
    global ready
    ready = False
    get_mapping_example()


def teleport_avatar(x=None, y=None, z=None):
    if x is not None and y is not None and z is not None:
        assert type(x) == int or type(x) == float, "Please a number"
        assert type(y) == int or type(y) == float, "Please a number"
        assert type(z) == int or type(z) == float, "Please a number"
        tdw.send_to_server({"$type": "teleport_avatar_to", "avatar_id": "uniqueid0", "env_id": 0,
                            "position": {"x": x, "y": y, "z": z}})
    else:
        tdw.send_to_server({"$type": "teleport_avatar_random", "avatar_id": "uniqueid0", "env_id": 0})


def teleport_object(tdw_object, obj_id, pos):
    tdw_object.communicate({"$type": "teleport_object", "id": obj_id, "position": pos})


def bend_arm(joint, angle, axis):
    # tdw.send_to_server(
    #     {"$type": "bend_arm_joint_by", "angle": angle, "avatar_id": "uniqueid0", "axis": axis, "env_id": 0,
    #      "joint": joint})
    tdw.send_to_server({"$type": "bend_arm_joint_to", "angle": angle, "avatar_id": "uniqueid0", "axis": axis, "env_id": 0, "joint": joint})



def apply_force_source_target(magnitude, source, target):
    tdw.send_to_server({"$type": "apply_force", "magnitude": magnitude,
                        "origin": source,
                        "target": target})


def get_mapping_example():
    """
    Example of how to receive ID and color maps.
    """

    tdw.send_to_server({"$type": "get_objects_data"})


def video_capture_init(directory):
    tdw.send_to_server({"$type": "init_video_capture", "avatar_id": "uniqueid0", "env_id": 0})
    # tdw.send_to_server({"$type": "set_temp_video_directory", "avatar_id": "uniqueid0", "directory": directory, "env_id": 0})


def start_video():
    tdw.send_to_server({"$type": "start_video_capture", "avatar_id": "uniqueid0", "env_id": 0})


def send_video():
    tdw.send_to_server({"$type": "send_video", "avatar_id": "uniqueid0", "env_id": 0})


def stop_video():
    tdw.send_to_server({"$type": "stop_video_capture", "avatar_id": "uniqueid0", "env_id": 0})


