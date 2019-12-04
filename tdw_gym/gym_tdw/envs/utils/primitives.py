from gym_tdw.envs.utils.object_utils import create_object
from gym_tdw.envs.utils.aux_utils import teleport_object, step_one_frame
from base64 import b64encode
import random


global_object_creator = {"x": -7.504, "y": 3, "z": -5.012}
global_ptr = 0
global_list = [0,1,2,1,0,0,0,0,0,2,1,0,2,1,0,1,0,1,0,2,1,0,2,0,0,2,1,1,0,2,1,0,1]


def create_highlight(tdw_object, x, z):
    with open("green.png", "rb") as f:
        image = b64encode(f.read()).decode("utf-8")
    painting_id = tdw_object.get_unique_id()
    painting_position = {"x": x, "y": 0.8324, "z": z}
    dimensions = {"x": 1, "y": 1}
    tdw_object.communicate([{"$type": "create_painting",
                             "position": painting_position,
                             "size": {"x": 0.5, "y": 0.5},
                             "euler_angles": {"x": 90, "y": 0, "z": 0},
                             "id": painting_id},
                            {"$type": "set_painting_texture",
                             "id": painting_id,
                             "dimensions": dimensions,
                             "image": image}
                            ])
    return painting_id


def create_goal_with_wall(tdw_object, x, z, enable_wall, highlight=False):
    if highlight:
        area_id = create_highlight(tdw_object, x, z)
        highlighted_areas = {}
        highlighted_areas[area_id] = "data"
    else:
        highlighted_areas = None


    # for side in enable_wall:
    #     if side == "bottom":
    #         thin_wall(tdw_object, tele_pos={"x": x, "y": 0.8749, "z": z-goal_bottom_diff}, rot={"x": 0, "y": 0, "z": 0})
    #     if side == "top":
    #         thin_wall(tdw_object, tele_pos={"x": x, "y": 0.8749, "z": z + diff}, rot={"x": 0, "y": 0, "z": 0})
    #     if side == "left":
    #         thin_wall(tdw_object, tele_pos={"x": x-diff, "y": 0.8749, "z": z}, rot={"x": 0, "y": 90, "z": 0})
    #     if side == "right":
    #         thin_wall(tdw_object, tele_pos={"x": x+diff, "y": 0.8749, "z": z}, rot={"x": 0, "y": 90, "z": 0})
    diff = 0.2388
    boundaries = {
        "z_top": z + diff,
        "z_bottom": z - diff,
        "x_left": x - diff,
        "x_right": x + diff
    }
    return highlighted_areas, boundaries


def create_wall_piece(tdw_object, wall_orientation, tele_pos):
    def _create_wall_piece(wall_scale, _pos, rot= {"x": 0, "y": 0, "z": 0}):
        global global_object_creator
        pos = {"x": global_object_creator["x"],
               "y": global_object_creator["y"],
               "z": global_object_creator["z"]}
        global_object_creator["x"] -= 1.5
        wall_id = tdw_object.add_object("prim_cube", position=pos, rotation=rot)
        tdw_object.communicate([
            {"$type": "scale_object", "id": wall_id, "scale_factor": wall_scale},
            # {"$type": "set_visual_material", "id": cube, "new_material_name": material_list[profile],
            #  "object_name": "PrimCube_0",
            #  "old_material_index": 0},
            {"$type": "set_kinematic_state", "id": wall_id, "is_kinematic": True, "use_gravity": False},
            {"$type": "teleport_object", "id": wall_id, "position": _pos}
        ])

    wall_length = 0.141
    corner_diff = 0.036
    thickness = 0.024
    if wall_orientation == "h":
        _create_wall_piece({"x": wall_length, "y": 0.1, "z": thickness}, tele_pos)
    elif wall_orientation == "v":
        _create_wall_piece({"x": thickness, "y": 0.1, "z": wall_length}, tele_pos)
    elif wall_orientation == "c1":
        # {"x": 0.105, "y": 0.1, "z":0.01}
        _pos1 = {"x": tele_pos["x"] + corner_diff, "y": tele_pos["y"], "z": tele_pos["z"] - corner_diff}
        _create_wall_piece({"x": 0.105, "y": 0.1, "z":thickness}, _pos1,  {"x": 0, "y": -45, "z": 0})

    elif wall_orientation == "c2":
        _pos1 = {"x": tele_pos["x"] - corner_diff, "y": tele_pos["y"], "z": tele_pos["z"] - corner_diff}
        _create_wall_piece({"x": 0.105, "y": 0.1, "z": thickness}, _pos1, {"x": 0, "y": 45, "z": 0})

    elif wall_orientation == "c3":
        _pos1 = {"x": tele_pos["x"] + corner_diff, "y": tele_pos["y"], "z": tele_pos["z"] + corner_diff}
        _create_wall_piece({"x": 0.105, "y": 0.1, "z": thickness}, _pos1, {"x": 0, "y": 45, "z": 0})

    elif wall_orientation == "c4":
        _pos1 = {"x": tele_pos["x"] - corner_diff, "y": tele_pos["y"], "z": tele_pos["z"] + corner_diff}
        _create_wall_piece({"x": 0.105, "y": 0.1, "z": thickness}, _pos1, {"x": 0, "y": -45, "z": 0})
    elif wall_orientation == "vb":
        _pos1 = {"x": tele_pos["x"], "y": tele_pos["y"], "z": tele_pos["z"] + corner_diff}
        _create_wall_piece({"x": thickness, "y": 0.1, "z": wall_length/2}, _pos1)
    elif wall_orientation == "vt":
        _pos1 = {"x": tele_pos["x"], "y": tele_pos["y"], "z": tele_pos["z"] - corner_diff}
        _create_wall_piece({"x": thickness, "y": 0.1, "z": wall_length/2}, _pos1)
    elif wall_orientation == "hr":
        _pos1 = {"x": tele_pos["x"] - corner_diff, "y": tele_pos["y"], "z": tele_pos["z"]}
        _create_wall_piece({"x": wall_length/2, "y": 0.1, "z": thickness}, _pos1)
    elif wall_orientation == "hl":
        _pos1 = {"x": tele_pos["x"] + corner_diff, "y": tele_pos["y"], "z": tele_pos["z"]}
        _create_wall_piece({"x": wall_length/2, "y": 0.1, "z": thickness}, _pos1)


def thin_wall(tdw_object, tele_pos, rot={"x": 0, "y": 0, "z": 0}):
    global global_object_creator
    pos = {"x": global_object_creator["x"],
           "y": global_object_creator["y"],
           "z": global_object_creator["z"]}
    global_object_creator["x"] -= 1.1
    wall_id = tdw_object.get_unique_id()
    tdw_object.communicate({"$type": "add_object",
     "env_id": 0,
     "model_name": "prim_cube",
     "position": pos,
     "rotation": rot,
     "id": wall_id})

    tdw_object.communicate([
        {"$type": "scale_object", "id": wall_id, "scale_factor": {"x": 0.01, "y": 0.1, "z": 0.141}},
        # {"$type": "set_visual_material", "id": cube, "new_material_name": material_list[profile],
        #  "object_name": "PrimCube_0",
        #  "old_material_index": 0},
        {"$type": "set_kinematic_state", "id": wall_id, "is_kinematic": True, "use_gravity": False},
        {"$type": "teleport_object", "id": wall_id, "position": tele_pos}
    ])
    return wall_id


def create_ramp(tdw_object, pos, type=1, rot={"x": 0, "y": 90, "z": 0}, scale={"x": 0.1, "y": 0.1, "z": 0.1}):
    global global_object_creator, global_ptr
    global_object_creator["x"] = global_object_creator["x"] - 1.1
    if type == 1:
        ramp = tdw_object.add_object("ramp_with_platform", position=dict(global_object_creator), rotation=rot)
        name = "ramp_with_platform"
    elif type == 2:
        ramp = tdw_object.add_object("ramp_with_platform_30", position=dict(global_object_creator), rotation=rot)
        name = "ramp_with_platform_30"
    else:
        ramp = tdw_object.add_object("ramp_with_platform_60", position=dict(global_object_creator), rotation=rot)
        name = "ramp_with_platform_60"
    tdw_object.communicate([
        {"$type": "scale_object", "id": ramp, "scale_factor": scale},
        {"$type": "teleport_object", "id": ramp, "position": pos},
        {"$type": "set_kinematic_state", "id": ramp, "is_kinematic": True, "use_gravity": False},
        {"$type": "set_visual_material", "id": ramp, "new_material_name": "plastic_microbead_grain_light",
         "object_name": name,
         "old_material_index": 0}
    ])
    return ramp


def create_main_sphere(tdw_object, tele_pos):
    sphere = create_object(tdw_object, "prim_sphere", {"x": -4.825, "y": 3, "z": -5.012})
    tdw_object.communicate(
        {"$type": "set_visual_material", "id": sphere, "new_material_name": "sls_plastic",
         "object_name": "PrimSphere_0",
         "old_material_index": 0, "quality": "med"})
    scale = {"x": 0.1, "y": 0.1, "z": 0.1}
    tdw_object.communicate({"$type": "scale_object", "id": sphere, "scale_factor": scale})
    teleport_object(tdw_object, sphere, tele_pos)
    tdw_object.communicate({"$type": "set_semantic_material", "id": sphere})
    tdw_object.communicate({"$type": "set_mass", "id": sphere, "mass": 7.0})
    tdw_object.communicate(
        {"$type": "set_physic_material", "id": sphere, "dynamic_friction": 0.1,
         "static_friction": 0.1,
         "bounciness": 0.1})
    return sphere


def create_target_sphere(tdw_object, pos, tele_pos):
    sphere = create_object(tdw_object, "prim_sphere", pos)
    scale = {"x": 0.1, "y": 0.1, "z": 0.1}
    tdw_object.communicate({"$type": "scale_object", "id": sphere, "scale_factor": scale})
    tdw_object.communicate(
        {"$type": "set_visual_material", "id": sphere, "new_material_name": "plastic_vinyl_glossy_yellow",
         "object_name": "PrimSphere_0",
         "old_material_index": 0})
    tdw_object.communicate([{"$type": "set_mass", "id": sphere, "mass": 1},
                           {"$type": "set_physic_material", "dynamic_friction": 0.1,
                            "static_friction": 0.1,
                            "bounciness": 0.9, "id": sphere}]
                           )
    teleport_object(tdw_object, sphere, tele_pos)
    return sphere


def create_push_sphere(tdw_object, pos, tele_pos):
    sphere = create_object(tdw_object, "prim_sphere", pos)
    scale = {"x": 0.1, "y": 0.1, "z": 0.1}
    tdw_object.communicate({"$type": "scale_object", "id": sphere, "scale_factor": scale})
    tdw_object.communicate(
        {"$type": "set_visual_material", "id": sphere, "new_material_name": "car_iridescent_paint",
         "object_name": "PrimSphere_0",
         "old_material_index": 0})
    tdw_object.communicate({"$type": "set_mass", "id": sphere, "mass": 1})
    teleport_object(tdw_object, sphere, tele_pos)
    return sphere


def create_cube(tdw_object, profile, pos, tele_pos):
    material_list = ["linen_viscose_classic_pattern",   "plastic_vinyl_glossy_orange", "polyester_sport_fleece_brushed"]
    mass = [10, 5, 100]
    params = [{"$type": "set_physic_material",  "dynamic_friction": 0.1,
         "static_friction": 0.1,
         "bounciness": 0.5},
              {"$type": "set_physic_material",  "dynamic_friction": 0.1,
               "static_friction": 0.1,
               "bounciness": 0.1},
              {"$type": "set_physic_material",  "dynamic_friction": 0.2,
               "static_friction": 0.2,
               "bounciness": 0.9}
              ]

    cube = tdw_object.add_object("prim_cube", position=pos, rotation={"x":0, "y":0, "z":0})

    tdw_object.communicate([

        {"$type": "set_mass", "id": cube, "mass": mass[profile]},
        {"$type": "scale_object", "id": cube, "scale_factor": {"x": 0.1, "y": 0.1, "z": 0.1}},
        {"$type": "set_visual_material", "id": cube, "new_material_name": material_list[profile],
         "object_name": "PrimCube_0",
         "old_material_index": 0},
        {"$type": "teleport_object", "id": cube, "position": tele_pos}
                        ])

    cube_params = params[profile]
    cube_params['id'] = cube
    tdw_object.communicate(cube_params)
    return cube


def create_cube_breakable(tdw_object, pos, tele_pos, scale={"x":0.1, "y":0.1, "z":0.1}):
    cube = tdw_object.add_object("prim_cone", position=pos, rotation={"x": 0, "y": 0, "z": 0})
    tdw_object.communicate([
        {"$type": "set_mass", "id": cube, "mass": 1},
        {"$type": "scale_object", "id": cube, "scale_factor": scale},
        {"$type": "set_visual_material", "id": cube, "new_material_name": "marble_griotte",
         "object_name": "prim_cone_0",
         "old_material_index": 0},
        {"$type": "teleport_object", "id": cube, "position": tele_pos}
    ])
    tdw_object.communicate({"$type": "set_physic_material", "bounciness": 1.0, "dynamic_friction": 1.0, "id": cube, "static_friction": 1.0})

    return cube


def create_wall(tdw_object, x, z, mode, side, length, gap=0.141, profile=None):
    global global_object_creator, global_ptr, global_list
    reset_param = {}
    if (global_ptr + length) > len(global_list):
        global_ptr = 0
    if profile is None:
        profile = [random.randint(0,1) for _ in range(length)]
        global_ptr += length
    idx = 0
    if mode == "h":
        for _ in range(length):
            global_object_creator["x"] = global_object_creator["x"] - 1.1
            pos = dict({"x": x, "y": 0.8749, "z": z})
            cube_id = create_cube(tdw_object, profile[idx], dict(global_object_creator), pos)
            reset_param[cube_id] = pos
            if side == "r":
                x = x + gap
            else:
                x = x - gap
            idx += 1
    if mode == "v":
        for _ in range(length):
            global_object_creator["x"] = global_object_creator["x"] - 1.1
            pos = dict({"x": x, "y": 0.8749, "z": z})
            cube_id = create_cube(tdw_object, profile[idx], dict(global_object_creator), pos)
            reset_param[cube_id] = pos
            if side == "u":
                z = z + gap
            else:
                z = z - gap
            idx += 1
    return reset_param


def create_cyl(tdw_object, profile, tele_pos, rot={"x": 0, "y": 0, "z": 90}, scale={"x": 0.1, "y": 0.1, "z": 0.1}):
    global global_object_creator, global_ptr
    global_object_creator["x"] = global_object_creator["x"] - 1.1
    material_list = ["linen_viscose_classic_pattern", "plastic_vinyl_glossy_orange", "polyester_sport_fleece_brushed"]
    mass = [100, 10, 50]
    params = [{"$type": "set_physic_material", "dynamic_friction": 0.1,
               "static_friction": 0.1,
               "bounciness": 0.5},
              {"$type": "set_physic_material", "dynamic_friction": 0.3,
               "static_friction": 0.3,
               "bounciness": 0.1},
              {"$type": "set_physic_material", "dynamic_friction": 0.2,
               "static_friction": 0.2,
               "bounciness": 0.9}
              ]
    cyl = tdw_object.add_object("prim_cyl", position=dict(global_object_creator), rotation=rot)

    tdw_object.communicate([
        {"$type": "set_mass", "id": cyl, "mass": mass[profile]},
        {"$type": "scale_object", "id": cyl, "scale_factor": scale},
        {"$type": "set_visual_material", "id": cyl, "new_material_name": material_list[profile],
         "object_name": "PrimCyl_0",
         "old_material_index": 0},
        {"$type": "teleport_object", "id": cyl, "position": tele_pos}
    ])

    cube_params = params[profile]
    cube_params['id'] = cyl
    tdw_object.communicate(cube_params)
    return cyl


def lever(tdw_object, pos, orientation="h", centering=0, side="r"):
    # create_cyl(1, pos, scale={"x": 0.09, "y": 0.3, "z": 0.09})
    # stopper1 = create_cyl(1, {"x": -3.990618, "y": 0.886, "z": -4.208851}, {"x": 0, "y": 0, "z": 0},
    #                       scale={"x": 0.01, "y": 0.06, "z": 0.01})
    # stopper2 = create_cyl(1, {"x": -3.990618, "y": 0.886, "z": -4.307}, {"x": 0, "y": 0, "z": 0},
    #                       scale={"x": 0.01, "y": 0.06, "z": 0.01})
    # tdw.send_to_server({"$type": "set_kinematic_state", "id": stopper1, "is_kinematic": True, "use_gravity": False})
    # tdw.send_to_server({"$type": "set_kinematic_state", "id": stopper2, "is_kinematic": True, "use_gravity": False})
    reset_params = {}
    gap = 0.0522
    stopper1_pos = dict(pos)
    stopper2_pos = dict(pos)
    if orientation == "h":
        cyl_id = create_cyl(tdw_object, 1, pos, rot={"x": 0, "y": 0, "z": 90}, scale={"x": 0.09, "y": 0.3, "z": 0.09})
        reset_params[cyl_id] = {
            "position": dict(pos),
            "rotation": {"x": 0, "y": 0, "z": 90}
        }
        stopper1_pos["z"] = stopper1_pos["z"] - gap
        stopper2_pos["z"] = stopper2_pos["z"] + gap
        if side == "r":
            stopper1_pos["x"] = stopper1_pos["x"] + centering
            stopper2_pos["x"] = stopper2_pos["x"] + centering
        else:
            stopper1_pos["x"] = stopper1_pos["x"] - centering
            stopper2_pos["x"] = stopper2_pos["x"] - centering
        stopper1 = create_cyl(tdw_object, 1, stopper1_pos, {"x": 0, "y": 0, "z": 0},
                              scale={"x": 0.01, "y": 0.06, "z": 0.01})
        stopper2 = create_cyl(tdw_object, 1, stopper2_pos, {"x": 0, "y": 0, "z": 0},
                              scale={"x": 0.01, "y": 0.06, "z": 0.01})
    if orientation == "v":
        cyl_id = create_cyl(tdw_object, 1, pos, rot={"x": 90, "y": 0, "z": 0}, scale={"x": 0.09, "y": 0.2, "z": 0.09})
        reset_params[cyl_id] = {
            "position": dict(pos),
            "rotation": {"x": 90, "y": 0, "z": 0}
        }
        stopper1_pos["x"] = stopper1_pos["x"] - gap
        stopper2_pos["x"] = stopper2_pos["x"] + gap
        if side == "r":
            stopper1_pos["z"] = stopper1_pos["z"] + centering
            stopper2_pos["z"] = stopper2_pos["z"] + centering
        else:
            stopper1_pos["z"] = stopper1_pos["z"] - centering
            stopper2_pos["z"] = stopper2_pos["z"] - centering
        stopper1 = create_cyl(tdw_object, 1, stopper1_pos, {"x": 0, "y": 0, "z": 0},
                              scale={"x": 0.01, "y": 0.06, "z": 0.01})
        stopper2 = create_cyl(tdw_object, 1, stopper2_pos, {"x": 0, "y": 0, "z": 0},
                              scale={"x": 0.01, "y": 0.06, "z": 0.01})

    tdw_object.communicate({"$type": "set_kinematic_state", "id": stopper1, "is_kinematic": True, "use_gravity": False})
    tdw_object.communicate({"$type": "set_kinematic_state", "id": stopper2, "is_kinematic": True, "use_gravity": False})
    return reset_params


def lever_gate(tdw_object, pos, orientation="h", inside=1):
    reset_params = {}
    gap = 0.0522
    stopper1_pos = dict(pos)
    stopper2_pos = dict(pos)
    centering = 0.22
    if orientation == "h":
        cyl_id = create_cyl(tdw_object, 1, pos, rot={"x": 0, "y": 0, "z": 90}, scale={"x": 0.09, "y": 0.25, "z": 0.09})
        reset_params[cyl_id] = {
            "position": dict(pos),
            "rotation": {"x": 0, "y": 0, "z": 90}
        }
        stopper1_pos["z"] = stopper1_pos["z"] - gap*inside
        stopper2_pos["z"] = stopper2_pos["z"] - gap*inside

        stopper1_pos["x"] = stopper1_pos["x"] + centering
        stopper2_pos["x"] = stopper2_pos["x"] - centering

        stopper1 = create_cyl(tdw_object, 1, stopper1_pos, {"x": 0, "y": 0, "z": 0},
                              scale={"x": 0.01, "y": 0.06, "z": 0.01})
        stopper2 = create_cyl(tdw_object, 1, stopper2_pos, {"x": 0, "y": 0, "z": 0},
                              scale={"x": 0.01, "y": 0.06, "z": 0.01})
    if orientation == "v":
        cyl_id = create_cyl(tdw_object, 1, pos, rot={"x": 90, "y": 0, "z": 0}, scale={"x": 0.09, "y": 0.25, "z": 0.09})
        reset_params[cyl_id] = {
            "position": dict(pos),
            "rotation": {"x": 90, "y": 0, "z": 0}
        }
        stopper1_pos["x"] = stopper1_pos["x"] - gap*inside
        stopper2_pos["x"] = stopper2_pos["x"] - gap*inside

        stopper1_pos["z"] = stopper1_pos["z"] + centering
        stopper2_pos["z"] = stopper2_pos["z"] - centering
        stopper1 = create_cyl(tdw_object, 1, stopper1_pos, {"x": 0, "y": 0, "z": 0},
                              scale={"x": 0.01, "y": 0.06, "z": 0.01})
        stopper2 = create_cyl(tdw_object, 1, stopper2_pos, {"x": 0, "y": 0, "z": 0},
                              scale={"x": 0.01, "y": 0.06, "z": 0.01})

    tdw_object.communicate({"$type": "set_kinematic_state", "id": stopper1, "is_kinematic": True, "use_gravity": False})
    tdw_object.communicate({"$type": "set_kinematic_state", "id": stopper2, "is_kinematic": True, "use_gravity": False})
    return reset_params


def lever_small(tdw_object, pos, orientation="h", centering=0, side="r", lower_stopper=True, upper_stopper=True):
    gap = 0.0522
    stopper1_pos = dict(pos)
    stopper2_pos = dict(pos)
    reset_params = {}
    if orientation == "h":

        cyl_id = create_cyl(tdw_object, 1, pos, rot={"x": 0, "y": 0, "z": 90}, scale={"x": 0.09, "y": 0.2, "z": 0.09})
        reset_params[cyl_id] = {
            "position": dict(pos),
            "rotation": {"x": 0, "y": 0, "z": 90}
        }
        stopper1_pos["z"] = stopper1_pos["z"] - gap
        stopper2_pos["z"] = stopper2_pos["z"] + gap
        if side == "r":
            stopper1_pos["x"] = stopper1_pos["x"] + centering
            stopper2_pos["x"] = stopper2_pos["x"] + centering
        else:
            stopper1_pos["x"] = stopper1_pos["x"] - centering
            stopper2_pos["x"] = stopper2_pos["x"] - centering
        if lower_stopper:
            stopper1 = create_cyl(tdw_object, 1, stopper1_pos, {"x": 0, "y": 0, "z": 0},
                                  scale={"x": 0.01, "y": 0.06, "z": 0.01})
        if upper_stopper:
            stopper2 = create_cyl(tdw_object, 1, stopper2_pos, {"x": 0, "y": 0, "z": 0},
                                  scale={"x": 0.01, "y": 0.06, "z": 0.01})
    if orientation == "v":
        cyl_id = create_cyl(tdw_object, 1, pos, rot={"x": 90, "y": 0, "z": 0}, scale={"x": 0.09, "y": 0.2, "z": 0.09})
        reset_params[cyl_id] = {
            "position": dict(pos),
            "rotation": {"x": 90, "y": 0, "z": 0}
        }
        stopper1_pos["x"] = stopper1_pos["x"] - gap
        stopper2_pos["x"] = stopper2_pos["x"] + gap
        if side == "r":
            stopper1_pos["z"] = stopper1_pos["z"] + centering
            stopper2_pos["z"] = stopper2_pos["z"] + centering
        else:
            stopper1_pos["z"] = stopper1_pos["z"] - centering
            stopper2_pos["z"] = stopper2_pos["z"] - centering
        if lower_stopper:
            stopper1 = create_cyl(tdw_object, 1, stopper1_pos, {"x": 0, "y": 0, "z": 0},
                                  scale={"x": 0.01, "y": 0.06, "z": 0.01})
        if upper_stopper:
            stopper2 = create_cyl(tdw_object, 1, stopper2_pos, {"x": 0, "y": 0, "z": 0},
                                  scale={"x": 0.01, "y": 0.06, "z": 0.01})
    if lower_stopper:
        tdw_object.communicate({"$type": "set_kinematic_state", "id": stopper1, "is_kinematic": True, "use_gravity": False})
    if upper_stopper:
        tdw_object.communicate({"$type": "set_kinematic_state", "id": stopper2, "is_kinematic": True, "use_gravity": False})
    return reset_params


def create_goal(tdw_object, x, z):
    gap = 0.11
    create_wall(tdw_object, x, z, "v", "d", 3, gap=gap, profile=[2]*3)
    create_wall(tdw_object, x + gap, z - 3*gap, "h", "r", 3, gap=gap,profile=[2] * 3)
    create_wall(tdw_object, x + 4*gap, z, "v", "d", 3, gap=gap, profile=[2] * 3)
    boundaries = {
        "z_top": z,
        "z_bottom": z - 0.2281,
        "x_left": x + 0.1012,
        "x_right": x + 0.3381
    }
    return boundaries


def create_breakable_wall(tdw_object, x, z, mode, side, length, gap=0.141, y=0.8749):
    reset_params = {}
    global global_object_creator, global_ptr, global_list
    idx = 0
    if mode == "h":
        for _ in range(length):
            global_object_creator["x"] = global_object_creator["x"] - 1.1
            pos = dict({"x": x, "y": 0.8332242, "z": z})
            wall_id = create_cube_breakable(tdw_object, dict(global_object_creator), pos, scale={"x": 0.1, "y": 0.1, "z": 0.1})

            reset_params[wall_id] = pos
            if side == "r":
                x = x + gap
            else:
                x = x - gap
            idx += 1
    if mode == "v":
        for _ in range(length):
            global_object_creator["x"] = global_object_creator["x"] - 1.1
            pos = dict({"x": x, "y": 0.8332242, "z": z})
            wall_id = create_cube_breakable(tdw_object, dict(global_object_creator), pos, scale={"x": 0.1, "y": 0.1, "z": 0.1})
            reset_params[wall_id] = pos
            if side == "u":
                z = z + gap
            else:
                z = z - gap
            idx += 1
    return reset_params


def create_cube_stack(tdw_object, stack_length, x, z, profiles=None):
    y = 0.8830635
    reset_params = {}
    global global_object_creator, global_ptr, global_list
    idx = 0
    for _ in range(stack_length):
        global_object_creator["x"] = global_object_creator["x"] - 1.1
        new_pos = dict(global_object_creator)
        new_pos["y"] = 0
        if profiles != None and len(profiles) == stack_length:
            pos = {"x": x, "y": y, "z": z}
            cube_id = create_cube(tdw_object, profiles[idx], dict(global_object_creator), pos)
            reset_params[cube_id] = pos
            idx += 1
        y = y + 0.09991
    step_one_frame(tdw_object, 100)
    return reset_params


def create_rectangle(tdw_object, pos, rot):
    global global_object_creator
    material_list = ["linen_viscose_classic_pattern", "plastic_vinyl_glossy_orange", "polyester_sport_fleece_brushed"]
    mass = [10, 5, 100]
    params = [{"$type": "set_physic_material", "dynamic_friction": 0.1,
               "static_friction": 0.1,
               "bounciness": 0.5},
              {"$type": "set_physic_material", "dynamic_friction": 0.1,
               "static_friction": 0.1,
               "bounciness": 0.1},
              {"$type": "set_physic_material", "dynamic_friction": 0.2,
               "static_friction": 0.2,
               "bounciness": 0.9}
              ]
    global_object_creator["x"] = global_object_creator["x"] - 1.1
    _pos = dict({"x": global_object_creator["x"], "y": 0.8332242, "z": global_object_creator["z"]})
    cube = tdw_object.add_object("prim_cube", position=_pos, rotation=rot)

    tdw_object.communicate([
        {"$type": "set_mass", "id": cube, "mass": mass[1]},
        {"$type": "scale_object", "id": cube, "scale_factor": {"x": 0.1, "y": 0.1, "z": 0.3}},
        {"$type": "set_visual_material", "id": cube, "new_material_name": material_list[1],
         "object_name": "PrimCube_0",
         "old_material_index": 0},
        {"$type": "teleport_object", "id": cube, "position": pos}
    ])

    cube_params = params[1]
    cube_params['id'] = cube
    tdw_object.communicate(cube_params)
    reset_params = {
        cube: {
            "position": dict(pos),
            "rotation": dict(rot)
        }
    }
    return reset_params
