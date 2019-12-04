
from gym_tdw.envs.utils.aux_utils import step_one_frame



def control_task_ramp_sol(objects):
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -17},
                        "id": objects["sphere"]})
    step_one_frame(45)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 13, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)

    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -40},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 25},
                        "id": objects["sphere"]})
    step_one_frame(35)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)

    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(30)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(120)

    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -10},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -25},
                        "id": objects["sphere"]})
    step_one_frame(100)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(120)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 25, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(30)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -30},
                        "id": objects["sphere"]})
    step_one_frame(30)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -20, "y": 0, "z": 20},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(150)
    return
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -7, "y": 0, "z": -30},
                        "id": objects["sphere"]})
    step_one_frame(70)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 8, "y": 0, "z": -25},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(3)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -10, "y": 0, "z": 10},
                        "id": objects["sphere"]})
    step_one_frame(45)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(100)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 20},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(80)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -8},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -20, "y": 0, "z": 7},
                        "id": objects["sphere"]})
    step_one_frame(18)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 30},
                        "id": objects["sphere"]})
    step_one_frame(100)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 20},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -30},
                        "id": objects["sphere"]})
    step_one_frame(8)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 10},
                        "id": objects["sphere"]})
    step_one_frame(22)
    # -4.442012
    # -4.441873
    # -4.441981
    # -4.441978
    # -4.44272
    # tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 20},
    #                     "id": objects["sphere"]})
    # step_one_frame(50)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(50)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 20, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 20, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(30)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 20, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 15},
                        "id": objects["sphere"]})
    step_one_frame(15)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -22, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -20, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(30)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)

    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -20, "y": 0, "z": 20},
                        "id": objects["sphere"]})
    step_one_frame(5)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -25},
                        "id": objects["sphere"]})
    step_one_frame(300)



    # tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -20, "y": 0, "z": 15},
    #                     "id": objects["sphere"]})
    # step_one_frame(27)
    # tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 11, "y": 0, "z": 20},
    #                     "id": objects["sphere"]})
    # step_one_frame(25)
    # tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 20},
    #                     "id": objects["sphere"]})
    # step_one_frame(50)
    # tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": 10},
    #                     "id": objects["sphere"]})
    # step_one_frame(15)
    # tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": -8},
    #                     "id": objects["sphere"]})
    # step_one_frame(18)
    # tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 20, "y": 0, "z": 0},
    #                     "id": objects["sphere"]})
    # step_one_frame(25)
    # tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 20, "y": 0, "z": 3},
    #                     "id": objects["sphere"]})
    # step_one_frame(25)


def control_task_ramp_over_wall_sol(objects):
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 35},
                        "id": objects["sphere"]})
    step_one_frame(100)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(22)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -30},
                        "id": objects["sphere"]})
    step_one_frame(50)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -10},
                        "id": objects["sphere"]})
    step_one_frame(17)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    control_task_ramp_over_wall_sol_reset(objects)
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 35},
                        "id": objects["sphere"]})
    step_one_frame(100)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(22)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -30},
                        "id": objects["sphere"]})
    step_one_frame(50)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -10},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)

    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -30, "y": 0, "z": 5},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)

    control_task_ramp_over_wall_sol_reset(objects)
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 35},
                        "id": objects["sphere"]})
    step_one_frame(100)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(22)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -30},
                        "id": objects["sphere"]})
    step_one_frame(50)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -10},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)

    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -30, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -30},
                        "id": objects["sphere"]})
    step_one_frame(35)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": -5},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 30, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(50)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": -10},
                        "id": objects["sphere"]})
    step_one_frame(10)


def control_task_ramp_over_wall_sol_reset(objects):
    for id_wall in objects["walls"]:
        tdw.send_to_server([{"$type": "stop_object", "id": id_wall[0]},
                            {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": 0, "z": 0},
                             "id": id_wall[0]},
                            {"$type": "teleport_object", "id": id_wall[0],
                             "position": id_wall[1]}
                            ])
    for sphere_id in objects["spheres"]:
        tdw.send_to_server([{"$type": "stop_object", "id": sphere_id[0]},
                           {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": 0, "z": 0},
                            "id": sphere_id[0]},
                           {"$type": "teleport_object", "id": sphere_id[0],
                            "position": sphere_id[1]}
                           ])


def control_task_stack_1_sol(objects):
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -7},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 20, "y": 0, "z": 5},
                        "id": objects["sphere"]})
    step_one_frame(25)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -40},
                        "id": objects["sphere"]})
    step_one_frame(30)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": -20},
                        "id": objects["sphere"]})
    step_one_frame(30)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -10, "y": 0, "z": 20},
                        "id": objects["sphere"]})
    step_one_frame(45)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 20, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(30)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -10, "y": 0, "z": -5},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -10, "y": 0, "z": -45},
                        "id": objects["sphere"]})
    step_one_frame(100)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": 10},
                        "id": objects["sphere"]})
    step_one_frame(50)


def _control_task2_sol(objects):
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -60, "y": 0, "z": -60},
                        "id": objects["sphere"]})
    step_one_frame(70)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -15, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -5, "y": 0, "z": 15},
                        "id": objects["sphere"]})
    step_one_frame(20)

    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 45, "y": 0, "z": -45},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 45, "y": 0, "z": 50},
                        "id": objects["sphere"]})
    step_one_frame(20)


def control_task2_sol(objects):
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -5, "y": 0, "z": -20},
                        "id": objects["sphere"]})
    step_one_frame(19)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 30, "y": 0, "z": -20},
                        "id": objects["sphere"]})
    step_one_frame(25)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 3, "y": 0, "z": 60},
                        "id": objects["sphere"]})
    step_one_frame(23)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -13},
                        "id": objects["sphere"]})
    step_one_frame(7)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 15, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(15)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -12, "y": 0, "z": -10},
                        "id": objects["sphere"]})
    step_one_frame(12)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 15},
                        "id": objects["sphere"]})
    step_one_frame(8)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -20, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(13)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -20},
                        "id": objects["sphere"]})
    step_one_frame(5)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 5, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 10},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(6)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(1)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -40},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "stop_object", "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 30, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(13)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -25, "y": 0, "z": -30},
                        "id": objects["sphere"]})
    step_one_frame(8)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -25, "y": 0, "z": 25},
                        "id": objects["sphere"]})
    step_one_frame(4)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 20, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(2)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -20, "y": 0, "z": -20},
                        "id": objects["sphere"]})
    step_one_frame(130)


def control_task_lever_sol(objects):
    # tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -3, "y": 0, "z": 20},
    #                     "id": objects["sphere"]})
    # step_one_frame(50)
    for i in range(5):
        tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -3, "y": 0, "z": 20},
                            "id": objects["sphere"]})
    step_one_frame(40)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -7, "y": 0, "z": 10},
                        "id": objects["sphere"]})
    step_one_frame(20)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 5, "y": 0, "z": -40},
                        "id": objects["sphere"]})
    step_one_frame(34)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -20, "y": 0, "z": 25},
                        "id": objects["sphere"]})
    step_one_frame(35)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 20, "y": 0, "z": -10},
                        "id": objects["sphere"]})
    step_one_frame(35)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -10, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(10)


def control_task_cyl_1_sol(objects):
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 80},
                        "id": objects["sphere"]})
    step_one_frame(50)


def control_task_wall_sol(objects):
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 50},
                        "id": objects["sphere"]})
    count = 0
    resolution = 4
    step_one_frame(10)
    reset_control_task_wall(objects)
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -10, "y": 0, "z": 50},
                        "id": objects["sphere"]})
    step_one_frame(10)
    reset_control_task_wall(objects)
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -15, "y": 0, "z": 50},
                        "id": objects["sphere"]})
    step_one_frame(10)
    reset_control_task_wall(objects)
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 10, "y": 0, "z": 50},
                        "id": objects["sphere"]})
    step_one_frame(10)
    reset_control_task_wall(objects)
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 15, "y": 0, "z": 50},
                        "id": objects["sphere"]})
    step_one_frame(10)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -50, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(10)
    # while True:
    #
    #
    #     if count % 4 == 0 :
    #         utils.objects_ready = False
    #         utils.get_mapping_example()
    #         while not utils.objects_ready:
    #             pass
    #         print(json.dumps(utils.object_data, indent=2))
    #     else:
    #         step_one_frame(1)
    #     if count > 5:
    #         break
    #     count += 1
    #     print(count)


def reset_control_task_wall(objects):
    tdw.send_to_server([{"$type": "stop_object", "id": objects["sphere"]}, {"$type": "teleport_object", "id": objects["sphere"], "position": {"x": -4.262, "y": 0.8749, "z": -5.248}}])
    gap = 0.11
    pos = {"x": -4.521, "y": 0.8332242, "z": -4.492}
    for id in objects["walls"]:
        tdw.send_to_server([{"$type": "stop_object", "id": id},
                            {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": 0, "z": 0},
                             "id": id},
                            {"$type": "teleport_object", "id": id,
                             "position": dict(pos)}
                            ])
        pos["x"] = pos["x"] + gap

    # tdw.send_to_server([{"$type": "stop_object", "id": objects["target"]},
    #                     {"$type": "teleport_object", "id": objects["target"],
    #                      "position": {"x": -4.46, "y": 0.8749, "z": -3.944}}])


def control_task1_sol(objects):
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 20, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(15)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -15, "y": 0, "z": 60},
                        "id": objects["sphere"]})
    step_one_frame(30)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -15, "y": 0, "z": 0},
                        "id": objects["sphere"]})
    step_one_frame(30)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -30, "y": 0, "z": -30},
                        "id": objects["sphere"]})
    step_one_frame(30)
    # tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -20, "y": 0, "z": 0},
    #                     "id": objects["sphere"]})
    # step_one_frame(5)
    # tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": 20},
    #                     "id": objects["sphere"]})
    # step_one_frame(5)


def stacked_cube_sphere_sol(objects):
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 0, "y": 0, "z": -15},
                        "id": objects["sphere"]})
    step_one_frame(10)

    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": 20, "y": 0, "z": 12},
                        "id": objects["sphere"]})
    step_one_frame(37)
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x": -17, "y": 0, "z": -80},
                        "id": objects["sphere"]})
    step_one_frame(100)



