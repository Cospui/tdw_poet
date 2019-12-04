from gym_tdw.envs.utils import primitives
from gym_tdw.envs.utils.aux_utils import step_one_frame
from gym_tdw.envs.utils.object_utils import create_object, teleport_object
import numpy as np
import random
import time
# Table grid size 16 (rows) x 8 (columns)

table_corners = (-4.681, -3.568)
# If table is at 0,0
table_corners = (-0.581, 1.0559)
# table_corners = (-4.7513, -3.5602)
global_object_creator = {"x": -7.504, "y": 3, "z": -5.012}
diff = 0.141

def render_puzzle(tdw_object, puzzle_array):
    global table_corners, global_object_creator, diff
    reset_params = {}
    model_names = {}
    main_sphere = None
    target_sphere = []
    highlighted_areas = {}
    puzzle_type = "non-goal"
    hit_puzzle = False
    push_puzzle = False
    wall_target_i = wall_target_j = None
    goal_boundaries = None
    break_ids = []
    push_sphere = []
    collision_detection_commands = []
    for i in range(16):
        wall_size = 0
        wall_start = None
        for j in range(9):
            if puzzle_array[i, j] == 1:
                if wall_size == 0:
                    wall_start = (i, j)
                wall_size += 1
                if j == 8:
                    walls_ = primitives.create_wall(tdw_object, table_corners[0] + wall_start[1] * diff,
                                                    table_corners[1] - wall_start[0] * diff, "h", "r", wall_size)
                    model_names.update({v: 'prim_cube' for v in walls_.keys()})
                    reset_params.update(walls_)
                    wall_size = 0
            else:
                if wall_size > 0:
                    walls_ = primitives.create_wall(tdw_object, table_corners[0] + wall_start[1]*diff,
                                                    table_corners[1] - wall_start[0]*diff, "h", "r", wall_size)
                    model_names.update({v: 'prim_cube' for v in walls_.keys()})
                    reset_params.update(walls_)
                    wall_size = 0
            if puzzle_array[i, j] == 2:
                pos = {"x": global_object_creator["x"], "y": global_object_creator["y"],
                       "z": global_object_creator["z"]}
                global_object_creator["x"] += 1.1
                obj_pos = {"x": table_corners[0] + j * diff, "y": 0.8749, "z": table_corners[1] - i * diff}
                tgt_sphere = primitives.create_target_sphere(tdw_object, pos, obj_pos)
                reset_params[tgt_sphere] = obj_pos
                model_names[tgt_sphere] = "prim_sphere"
                target_sphere.append(tgt_sphere)
                hit_puzzle = True
                collision_detection_commands.append({"$type": "set_object_collision_detection_mode",
                                                     "id": tgt_sphere,
                                                     "mode": "continuous_dynamic"})
            if puzzle_array[i, j] == 15:
                pos = {"x": global_object_creator["x"], "y": global_object_creator["y"],
                       "z": global_object_creator["z"]}
                global_object_creator["x"] += 1.1
                obj_pos = {"x": table_corners[0] + j * diff, "y": 0.8749, "z": table_corners[1] - i * diff}
                tgt_sphere = primitives.create_push_sphere(tdw_object, pos, obj_pos)
                reset_params[tgt_sphere] = obj_pos
                model_names[tgt_sphere] = "prim_sphere"
                push_sphere.append(tgt_sphere)
                push_puzzle = True
            if puzzle_array[i, j] == 3:
                global_object_creator["x"] += 1.1
                obj_pos = {"x": table_corners[0] + j * diff, "y": 0.8749, "z": table_corners[1] - i * diff}
                main_sphere = primitives.create_main_sphere(tdw_object, tele_pos=obj_pos)
                model_names[main_sphere] = "prim_sphere"
                reset_params[main_sphere] = obj_pos
            if puzzle_array[i, j] == 4:
                global_object_creator["x"] += 1.1
                obj_pos = {"x": table_corners[0] + j * diff, "y": 0.8749, "z": table_corners[1] - i * diff}
                break_wall = primitives.create_breakable_wall(tdw_object, obj_pos["x"], obj_pos["z"], "h", "r", 1)
                break_ids.extend(list(break_wall.keys()))
                reset_params.update(break_wall)
            if puzzle_array[i, j] == 8:
                global_object_creator["x"] += 1.1
                obj_pos = {"x": table_corners[0] + j * diff, "y": 0.8749, "z": table_corners[1] - i * diff}
                enable_walls = []
                if puzzle_array[i-2, j-2:j+3].sum() == 30:
                    enable_walls.append("top")
                if puzzle_array[i-2:i+3, j-2].sum() == 30:
                    enable_walls.append("left")
                if puzzle_array[i-2:i+3, j+2].sum() == 30:
                    enable_walls.append("right")
                if puzzle_array[i+2, j-2:j+3].sum() == 30:
                    enable_walls.append("bottom")
                highlight_id, goal_boundaries = primitives.create_goal_with_wall(tdw_object, obj_pos["x"], obj_pos["z"], enable_wall=enable_walls,
                                                 highlight=True)
                puzzle_type = "goal"
                highlighted_areas.update(highlight_id)

                # Find inside ramp center
                for _i in range(i - 1, i + 2):
                    for _j in range(j - 1, j + 2):
                        if puzzle_array[_i, _j] == 14:
                            break
                    if puzzle_array[_i, _j] == 14:
                        break
                # Put the ramp inside the walled target
                obj_pos = {"x": table_corners[0] + _j * diff, "y": 0.8304, "z": table_corners[1] - _i * diff}
                if puzzle_array[_i, _j] == 14:
                    rot = {"x": 0, "y": 90, "z": 0}

                    if _i + 1 < 16:
                        if puzzle_array[_i + 1, _j] == 9:
                            rot = {"x": 0, "y": 270, "z": 0}
                            obj_pos["x"] += 0.101
                            obj_pos["z"] -= 0.081
                    if _i - 1 > -1:
                        if puzzle_array[_i - 1, _j] == 9:
                            rot = {"x": 0, "y": 90, "z": 0}
                            obj_pos["x"] -= 0.11
                            obj_pos["z"] += 0.076
                    if _j + 1 < 9:
                        if puzzle_array[_i, _j + 1] == 9:
                            print("Here rgiht now")
                            rot = {"x": 0, "y": 180, "z": 0}
                            obj_pos["x"] += 0.081
                            obj_pos["z"] += 0.106
                    if _j - 1 > -1:
                        if puzzle_array[_i, _j - 1] == 9:
                            rot = {"x": 0, "y": 0, "z": 0}
                            obj_pos["x"] -= 0.076
                            obj_pos["z"] -= 0.106
                    primitives.create_ramp(tdw_object, obj_pos, rot=rot)

            if puzzle_array[i, j] == 13:
                global_object_creator["x"] += 1.1
                obj_pos = {"x": table_corners[0] + j * diff, "y": 0.8749, "z": table_corners[1] - i * diff}
                enable_walls = []
                enable_lever_gate = []
                wall_start_i = 0 if i - 1 < 0 else i - 1
                wall_end_i = 16 if i + 2 > 16 else i + 2
                wall_start_j = 0 if j - 1 < 0 else j - 1
                wall_end_j = 9 if j + 2 > 9 else j + 2

                if i != 1:
                    if puzzle_array[wall_start_i-1, wall_start_j:wall_end_j].sum() == 18:
                        enable_walls.append("top")
                    elif puzzle_array[wall_start_i-1, wall_start_j:wall_end_j].sum() == 30:
                        enable_lever_gate.append("top")
                if j != 1:
                    if puzzle_array[wall_start_i: wall_end_i, wall_start_j-1].sum() == 18:
                        enable_walls.append("left")
                    elif puzzle_array[wall_start_i: wall_end_i, wall_start_j-1].sum() == 30:
                        enable_lever_gate.append("left")
                if j != 7:
                    if puzzle_array[wall_start_i:wall_end_i, wall_end_j].sum() == 18:
                        enable_walls.append("right")
                    elif puzzle_array[wall_start_i:wall_end_i, wall_end_j].sum() == 30:
                        enable_lever_gate.append("right")
                if i != 14:
                    if puzzle_array[wall_end_i, wall_start_j:wall_end_j].sum() == 18:
                        enable_walls.append("bottom")
                    elif puzzle_array[wall_end_i, wall_start_j:wall_end_j].sum() == 30:
                        enable_lever_gate.append("bottom")
                # Create walled target
                # _, _ = primitives.create_goal_with_wall(tdw_object, obj_pos["x"], obj_pos["z"], enable_wall=enable_walls, highlight=False)


                diff_ = 0.28322
                if "left" in enable_lever_gate:
                    reset_params.update(primitives.lever_gate(tdw_object, {"x": obj_pos["x"]-diff_, "y": 0.8749, "z": obj_pos["z"]}, "v",  inside=1))
                if "right" in enable_lever_gate:
                    reset_params.update(primitives.lever_gate(tdw_object, {"x": obj_pos["x"]+diff_, "y": 0.8749, "z": obj_pos["z"]}, "v",  inside=-1))
                if "top" in enable_lever_gate:
                    reset_params.update(primitives.lever_gate(tdw_object, {"x": obj_pos["x"], "y": 0.8749, "z": obj_pos["z"]+diff_}, "h",  inside=1))
                if "bottom" in enable_lever_gate:
                    reset_params.update(primitives.lever_gate(tdw_object, {"x": obj_pos["x"], "y": 0.8749, "z": obj_pos["z"]-diff_}, "h", inside=-1))
                # Find inside ramp center
                for _i in range(i - 1, i + 2):
                    for _j in range(j - 1, j + 2):
                        if puzzle_array[_i, _j] == 14:
                            break
                    if puzzle_array[_i, _j] == 14:
                        break
                # Put the ramp inside the walled target
                obj_pos = {"x": table_corners[0] + _j * diff, "y": 0.8304, "z": table_corners[1] - _i * diff}
                if puzzle_array[_i, _j] == 14:
                    rot = {"x": 0, "y": 90, "z": 0}

                    if _i + 1 < 16:
                        if puzzle_array[_i+1, _j] == 9:
                            rot = {"x": 0, "y": 270, "z": 0}
                            obj_pos["x"] += 0.101
                            obj_pos["z"] -= 0.081
                    if _i - 1 > -1:
                        if puzzle_array[_i-1, _j] == 9:
                            rot = {"x": 0, "y": 90, "z": 0}
                            obj_pos["x"] -= 0.11
                            obj_pos["z"] += 0.076
                    if _j + 1 < 9:
                        if puzzle_array[_i, _j+1] == 9:
                            print("Here rgiht now")
                            rot = {"x": 0, "y": 180, "z": 0}
                            obj_pos["x"] += 0.081
                            obj_pos["z"] += 0.106
                    if _j - 1 > -1:
                        if puzzle_array[_i, _j-1] == 9:
                            rot = {"x": 0, "y": 0, "z": 0}
                            obj_pos["x"] -= 0.076
                            obj_pos["z"] -= 0.106
                    primitives.create_ramp(tdw_object, obj_pos, rot=rot)
            if puzzle_array[i, j] == 11:
                global_object_creator["x"] += 1.1
                # Check ramp orientation
                if not wall_target_i and not wall_target_j:
                    for _i in range(16):
                        for _j in range(9):
                            if puzzle_array[_i, _j] == 13 or puzzle_array[_i, _j] == 8:
                                wall_target_i = _i
                                wall_target_j = _j
                                break
                obj_pos = {"x": table_corners[0] + j * diff, "y": 0.8304, "z": table_corners[1] - i * diff}
                correction_offset = 0.09
                # Top of the goal
                if i < wall_target_i and j == wall_target_j:
                    rot = {"x": 0, "y": 90, "z": 0}
                    obj_pos["z"] -= correction_offset
                # Bottom of the goal
                elif i > wall_target_i and j == wall_target_j:
                    rot = {"x": 0, "y": 270, "z": 0}
                    obj_pos["z"] += correction_offset
                # Left of the goal
                elif j < wall_target_j and i == wall_target_i:
                    rot = {"x": 0, "y": 0, "z": 0}
                    obj_pos["x"] += correction_offset
                # Right of the goal
                elif j > wall_target_j and i == wall_target_i:
                    rot = {"x": 0, "y": 180, "z": 0}
                    obj_pos["z"] -= correction_offset
                primitives.create_ramp(tdw_object, obj_pos, rot=rot)

            # Insert Wall piece
            if puzzle_array[i, j] == 6:
                render_wall_piece(tdw_object, puzzle_array, i, j)
            # Insert Cube stack
            if puzzle_array[i,j] == 17:
                primitives.create_cube_stack(tdw_object, 2, table_corners[0] + j * diff, table_corners[1] - i * diff, profiles=[1]*2)
                tele_pos = {"x": table_corners[0] + j * diff, "y": 1.0835, "z": table_corners[1] - i * diff}
                pos = {"x": global_object_creator["x"], "y": global_object_creator["y"],
                       "z": global_object_creator["z"]}
                global_object_creator["x"] += 1.1
                tgt_sphere = primitives.create_target_sphere(tdw_object, pos, tele_pos)
                reset_params[tgt_sphere] = tele_pos
                model_names[tgt_sphere] = "prim_sphere"
                target_sphere.append(tgt_sphere)
            # Insert rectangle
            if puzzle_array[i, j] == 19:
                rot = None
                if 0 < i < 15:
                    if puzzle_array[i-1, j] == 18 and puzzle_array[i+1, j] == 18:
                        rot = {"x": 0, "y": 0, "z": 0}
                if 0 < j < 8:
                    if puzzle_array[i, j-1] == 18 and puzzle_array[i, j+1] == 18:
                        rot = {"x": 0, "y": 90, "z": 0}
                if rot:
                    obj_pos = {"x": table_corners[0] + j * diff, "y": 0.8304, "z": table_corners[1] - i * diff}
                    rect = primitives.create_rectangle(tdw_object, obj_pos, rot)
                    model_names.update({v: 'prim_cube' for v in rect.keys()})
                    reset_params.update(rect)

    if collision_detection_commands:
        tdw_object.communicate(collision_detection_commands)
    return_object = {"sphere": main_sphere,
                "target_spheres": target_sphere,
                "push_spheres": push_sphere,
                "reset_params": reset_params,
                "model_names": model_names,
                "highlighted_areas": highlighted_areas}
    if push_puzzle and hit_puzzle:
        puzzle_type = "hybrid"
    elif hit_puzzle:
        puzzle_type = "non-goal"
    else:
        puzzle_type = "goal"
    if goal_boundaries:
        return_object["goal_boundaries"] = goal_boundaries
    if break_ids:
        return_object["walls"] = break_ids
    return return_object, puzzle_type


def render_wall_piece(tdw_object, puzzle_array, i, j):
    global table_corners, global_object_creator, diff
    wall_orientation = None
    # Check wall orientation
    wall_left = False
    wall_up = False
    wall_below = False
    wall_right = False
    no_of_neighbor_wall = 0
    if j > 0:
        if puzzle_array[i, j - 1] == 6:
            wall_left = True
            no_of_neighbor_wall += 1
    if i < 15:
        if puzzle_array[i + 1, j] == 6:
            wall_below = True
            no_of_neighbor_wall += 1
    if j < 8:
        if puzzle_array[i, j+1] == 6:
            wall_right = True
            no_of_neighbor_wall += 1
    if i > 0:
        if puzzle_array[i - 1, j] == 6:
            wall_up = True
            no_of_neighbor_wall += 1

    # Is it corner piece ?
    if wall_up and wall_left:
        wall_orientation = "c4"
    elif wall_up and wall_right:
        wall_orientation = "c3"
    elif wall_below and wall_left:
        wall_orientation = "c2"
    elif wall_below and wall_right:
        wall_orientation = "c1"
    elif wall_up and not (wall_left or wall_right or wall_below):
        wall_orientation = "vb"
    elif wall_right and not (wall_up or wall_left or wall_below):
        wall_orientation = "hl"
    elif wall_left and not (wall_up or wall_right or wall_below):
        wall_orientation = "hr"
    elif wall_below and not (wall_up or wall_right or wall_left):
        wall_orientation = "vt"
    # Is it horizontal ?
    elif wall_left or wall_right:
        wall_orientation = "h"
    # Is it vertical ?
    elif wall_up or wall_below:
        wall_orientation = "v"

    obj_pos = {"x": table_corners[0] + j * diff, "y": 0.883, "z": table_corners[1] - i * diff}
    primitives.create_wall_piece(tdw_object, wall_orientation, obj_pos)
