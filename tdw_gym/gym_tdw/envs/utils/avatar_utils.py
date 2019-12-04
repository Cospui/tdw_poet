import TDW.tdw as tdw


def rotate_avatar(avatar, angle):
    tdw.send_to_server({"$type": "rotate_avatar_by", "avatar_id": avatar, "env_id": 0, "rotation": int(angle)})


def create_avatar(avatar_name, position, id):
    tdw.send_to_server([{"$type": "create_avatar", "type": avatar_name, "id": id},
                        {"$type": "teleport_avatar_to", "avatar_id": id, "env_id": 0,
                         "position": position}
                        ])


def set_avatar_output(image_toggle, avatar_id):
    tdw.send_to_server({"$type": "set_avatar_output", "avatar_id": avatar_id, "env_id": 0, "images": image_toggle,
                         "object_info": True, "avatar_info": True, "child_info": True,
                         "collision_info": True, "sensors_info": True})


def set_pass_mark(avatar_id, pass_marks=["_img"]):
    tdw.send_to_server({"$type": "set_pass_masks", "avatar_id": avatar_id,
                                           "pass_masks": pass_marks})


def rotate_senor_container(axis, angle, avatar_id):
    tdw.send_to_server({"$type": "rotate_sensor_container_by", "angle": angle, "avatar_id": avatar_id, "axis": axis,
                        "env_id": 0})


def bend_arm(joint, angle, axis, avatar_id):
    tdw.send_to_server({"$type": "bend_arm_joint_to", "angle": angle, "avatar_id": avatar_id, "axis": axis,
                        "env_id": 0, "joint": joint})


def toggle_sensor_container(avatar_id, conainer_type):
    tdw.send_to_server({"$type": "toggle_sensor", "avatar_id": avatar_id, "env_id": 0, "sensor_name": conainer_type})
