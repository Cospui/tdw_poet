import random

global_object_creator = {"x": -7.504, "y": 3, "z": -5.012}

material_list = ["linen_viscose_classic_pattern",   "plastic_vinyl_glossy_orange", "polyester_sport_fleece_brushed"]
profiles = {}


def create_objects(experiment_parameters):
    _create_objects(experiment_parameters)
    teleport_objects(experiment_parameters)


def teleport_objects(experiment_parameters):

    for obj in experiment_parameters["objects"]:
        teleport_object(experiment_parameters[obj]["id"], {"x": experiment_parameters[obj]["position"]["x"],
                                                     "y": experiment_parameters[obj]["position"]["y"],
                                                     "z": experiment_parameters[obj]["position"]["z"]})


def create_main_sphere(tele_pos):
    sphere = create_object("prim_sphere", {"x": -4.825, "y": 3, "z": -5.012})
    tdw.send_to_server(
        {"$type": "set_visual_material", "id": sphere, "new_material_name": "sls_plastic",
         "object_name": "PrimSphere_0",
         "old_material_index": 0})
    scale = 0.1
    tdw.send_to_server({"$type": "scale_object", "id": sphere, "scale_factor": scale})
    teleport_object(sphere, tele_pos)
    tdw.send_to_server({"$type": "set_mass", "id": sphere, "mass": 7.0})
    tdw.send_to_server(
        {"$type": "set_physic_material", "id": sphere, "dynamic_friction": 0.1,
         "static_friction": 0.1,
         "bounciness": 0.1})
    return sphere


def create_target_sphere(pos, tele_pos):
    sphere = create_object("prim_sphere", pos)
    tdw.send_to_server({"$type": "scale_object", "id": sphere, "scale_factor": 0.1})
    tdw.send_to_server(
        {"$type": "set_visual_material", "id": sphere, "new_material_name": "plastic_vinyl_glossy_yellow",
         "object_name": "PrimSphere_0",
         "old_material_index": 0})
    teleport_object(sphere, tele_pos)
    return sphere


def set_profile(experiment_params):
    global material_list, profiles
    profiles[1] = {"material":None}
    profiles[1]["material"] = material_list[0]
    profiles[1]["mass"] = int(experiment_params["obj1"])
    profiles[1]["physical_material"] = {"$type": "set_physic_material",  "dynamic_friction": 0.1,
         "static_friction": 0.1,
         "bounciness": 0.5}

    profiles[2] = {"material": None}
    profiles[2]["material"] = material_list[1]
    profiles[2]["mass"] = int(experiment_params["obj2"])
    profiles[2]["physical_material"] = {"$type": "set_physic_material", "dynamic_friction": 0.1,
                                          "static_friction": 0.1,
                                          "bounciness": 0.5}

    profiles[3] = {"material": None}
    profiles[3]["material"] = material_list[2]
    profiles[3]["mass"] = int(experiment_params["obj3"])
    profiles[3]["physical_material"] = {"$type": "set_physic_material", "dynamic_friction": 0.1,
                                          "static_friction": 0.1,
                                          "bounciness": 0.9}


def create_cube(profile, tele_pos):
    global global_object_creator, profiles
    pos = global_object_creator
    global_object_creator["x"] -= 1.1

    cube = tdw.get_unique_id()
    tdw.send_to_server({"$type": "add_objects", "env_id": 0, "objects":
            [{"name": "prim_cube", "position": pos, "orientation":
                {"x":0, "y":0, "z":0}, "id": cube}
             ]
         })
    tdw.send_to_server([

        {"$type": "set_mass", "id": cube, "mass": profiles[profile]["mass"]},
        {"$type": "scale_object", "id": cube, "scale_factor": 0.1},
        {"$type": "set_visual_material", "id": cube, "new_material_name": profiles[profile]["material"],
         "object_name": "PrimCube_0",
         "old_material_index": 0},
        {"$type": "teleport_object", "id": cube, "position": tele_pos}
                        ])

    cube_params = profiles[profile]["physical_material"]
    cube_params['id'] = cube
    tdw.send_to_server(cube_params)
    return cube


def generate_position(object_positions=None):
    x = random.uniform(-3.565, -4.688)
    z = random.uniform(-3.588, -5.641)
    y = 0.8749
    if object_positions:
        while overlap(x,z, object_positions):
            x = random.uniform(-3.565, -4.688)
            z = random.uniform(-3.588, -5.641)
    return {"x": x, "y": y, "z": z}


def overlap(x, z, object_positions):
    for pos in object_positions:
        if abs(pos["x"] - x) < 0.111 and abs(pos["z"] - z) < 0.111:
            return True
    return False


def _create_objects(experiment_parameters):
    position_list = [{"x": -4.825, "y": 3, "z": -5.012}, {"x": -3.51, "y": 3, "z": -5.012},
                     {"x": -2.179, "y": 3, "z": -5.012}, {"x": -0.787, "y": 3, "z": -5.012}]
    for i, obj in enumerate(experiment_parameters["objects"]):
        experiment_parameters[obj]["id"] = create_object("prim_sphere", position_list[i])
        tdw.send_to_server({"$type": "scale_object", "id": experiment_parameters[obj]["id"], "scale_factor": 0.1})
    return experiment_parameters


def teleport_object(obj_id, pos):
    tdw.send_to_server({"$type": "teleport_object", "id": obj_id, "position": pos})


def create_object(name, pos, rot={"x": 0.0, "y": 0.0, "z": 0.0}):
    obj_list_f = tdw.get_unique_id()
    tdw.send_to_server({"$type": "add_objects", "env_id": 0, "objects":
        [{"name": name, "position": pos, "orientation":
            rot, "id": obj_list_f}]})
    return obj_list_f

def create_object(tdw_object, name, pos, rot={"x": 0.0, "y": 0.0, "z": 0.0}):
    obj_list_f = tdw_object.get_unique_id()
    tdw_object.communicate({"$type": "add_object",
                      "env_id": 0,
                      "model_name": name,
                      "position": pos,
                      "rotation": rot,
                      "id": obj_list_f})

    return obj_list_f


def apply_force_object(object, force):
    tdw.send_to_server({"$type": "apply_force_to_object", "force": {"x":int(force[0]), "y":0, "z":int(force[1])}, "id": object})


def set_physical_parameters(experiment_parameters):
    for obj in experiment_parameters["objects"]:
        tdw.send_to_server({"$type": "set_mass", "id": experiment_parameters[obj]["id"],
                            "mass": experiment_parameters[obj]["mass"]})
        tdw.send_to_server(
            {"$type": "set_physic_material", "id": experiment_parameters[obj]["id"],
             "dynamic_friction": experiment_parameters[obj]["friction"],
             "static_friction": experiment_parameters[obj]["friction"],
             "bounciness": experiment_parameters[obj]["bounciness"]})


def apply_materials(experiment_parameters):
    material_list = [ "synthetic_eco_lor_french_terry", "sls_plastic", "metallic_car_paint", "carbon_fiber_twill_weave"]
    # random.shuffle(material_list)
    for i, obj in enumerate(experiment_parameters["objects"]):
        tdw.send_to_server(
            {"$type": "set_visual_material", "id": experiment_parameters[obj]["id"], "new_material_name": material_list[i],
             "object_name": "PrimSphere_0",
             "old_material_index": 0})


def stop_objects(experiment_parameters):
    for obj in experiment_parameters["objects"]:
        tdw.send_to_server({"$type": "stop_object", "id": experiment_parameters[obj]["id"]})