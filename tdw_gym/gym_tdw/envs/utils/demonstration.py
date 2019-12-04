from gym_tdw.envs.utils.object_utils import create_object, teleport_object
import numpy as np
import math
import random
from gym_tdw.envs.utils.aux_utils import step_one_frame
import gym_tdw.envs.utils.aux_utils as utils
import os
import pickle
from gym_tdw.envs.utils import primitives


def demonstration_function():
    material_list = ["polyester_sport_fleece_brushed", "sls_plastic", "metallic_car_paint", "carbon_fiber_twill_weave",
                     "plastic_vinyl_glossy_orange", "plastic_hammered", "linen_viscose_classic_pattern", "leather_pu_perforated_dots"]
    # material_list = ["linen_viscose_classic_pattern", "plastic_vinyl_glossy_orange", "polyester_sport_fleece_brushed"]

    sphere = create_object("prim_sphere", {"x": -4.825, "y": 3, "z": -5.012})
    tdw.send_to_server(
        {"$type": "set_visual_material", "id": sphere, "new_material_name": material_list[1],
         "object_name": "PrimSphere_0",
         "old_material_index": 0})
    scale = 0.1
    tdw.send_to_server({"$type": "scale_object", "id": sphere, "scale_factor": scale})
    sphere_pos = {"x": -4.46, "y": 0.8749, "z": -3.944}
    teleport_object(sphere, sphere_pos)
    tdw.send_to_server(
        {"$type": "set_physic_material", "id": sphere, "dynamic_friction": 0.1,
         "static_friction": 0.1,
         "bounciness": 0.1})

    cube = create_object("prim_cube", {"x": -7.504, "y": 3, "z": -5.012})
    cube2 = create_object("prim_cube", {"x": -10.504, "y": 3, "z": -5.012})
    cube3 = create_object("prim_cube", {"x": -13.504, "y": 3, "z": -5.012})
    # cube4 = create_object("prim_cube", {"x": -15.504, "y": 3, "z": -5.012})
    # cube5 = create_object("prim_cube", {"x": -17.504, "y": 3, "z": -5.012})
    # cube6 = create_object("prim_cube", {"x": -19.504, "y": 3, "z": -5.012})
    sphere5 = create_object("prim_sphere", {"x": -19.787, "y": 3, "z": -5.012})
    tdw.send_to_server(
        {"$type": "set_visual_material", "id": cube, "new_material_name": material_list[0],
         "object_name": "PrimCube_0",
         "old_material_index": 0})
    tdw.send_to_server(
        {"$type": "set_visual_material", "id": cube2, "new_material_name": material_list[6],
         "object_name": "PrimCube_0",
         "old_material_index": 0})
    tdw.send_to_server(
        {"$type": "set_visual_material", "id": cube3, "new_material_name": material_list[4],
         "object_name": "PrimCube_0",
         "old_material_index": 0})
    tdw.send_to_server(
        {"$type": "set_visual_material", "id": sphere5, "new_material_name": "plastic_vinyl_glossy_yellow",
         "object_name": "PrimSphere_0",
         "old_material_index": 0})
    tdw.send_to_server({"$type": "set_mass", "id": cube, "mass": 10000.0})
    tdw.send_to_server({"$type": "set_mass", "id": cube2, "mass": 50.0})
    tdw.send_to_server({"$type": "set_mass", "id": cube3, "mass": 10.0})
    tdw.send_to_server({"$type": "set_mass", "id": sphere, "mass": 7.0})
    # tdw.send_to_server({"$type": "set_mass", "id": cube4, "mass": 10000.0})
    tdw.send_to_server({"$type": "scale_object", "id": cube, "scale_factor": scale})
    tdw.send_to_server({"$type": "scale_object", "id": cube2, "scale_factor": scale})
    tdw.send_to_server({"$type": "scale_object", "id": cube3, "scale_factor": scale})
    # tdw.send_to_server({"$type": "scale_object", "id": cube4, "scale_factor": scale})
    # tdw.send_to_server({"$type": "scale_object", "id": cube5, "scale_factor": scale})
    # tdw.send_to_server({"$type": "scale_object", "id": cube6, "scale_factor": scale})
    tdw.send_to_server({"$type": "scale_object", "id": sphere5, "scale_factor": scale})

    cube_pos = {"x": -3.997, "y": 0.8749, "z": -5.238}
    cube2_pos = {"x": -3.864, "y": 0.8749, "z": -4.023}
    cube3_pos = {"x": -4.448, "y": 0.8749, "z": -4.604}
    # cube4_pos = {"x": -4.174, "y": 0.8749, "z": -5.014}
    teleport_object(cube, cube_pos)
    teleport_object(cube2, cube2_pos)
    teleport_object(cube3, cube3_pos)
    # teleport_object(cube4, cube4_pos)
    # teleport_object(cube5, {"x": -4.515, "y": 0.8749, "z": -5.062})
    # teleport_object(cube6, {"x": -3.884, "y": 0.8749, "z": -4.99})
    # teleport_object(sphere5, {"x": -4.272, "y": 0.8749, "z": -5.494})

    tdw.send_to_server(
        {"$type": "set_physic_material", "id": cube, "dynamic_friction": 0.1,
         "static_friction": 0.1,
         "bounciness": 0.5})
    tdw.send_to_server(
        {"$type": "set_physic_material", "id": cube2, "dynamic_friction": 0.2,
         "static_friction": 0.2,
         "bounciness": 0.9})
    tdw.send_to_server(
        {"$type": "set_physic_material", "id": cube3, "dynamic_friction": 0.3,
         "static_friction": 0.3,
         "bounciness": 0.1})


    tdw.send_to_server(
        {"$type": "set_physic_material", "id": sphere, "dynamic_friction": 0.9,
         "static_friction": 0.9,
         "bounciness": 0.1})
    target_sphere_pos = {"x": -3.834, "y": 0.8749, "z": -4.668}
    target_sphere = primitives.create_target_sphere({"x": -19.787, "y": 3, "z": -5.012}, target_sphere_pos)
    return_objects = {
        "sphere": {
            "id": sphere,
            "pos": sphere_pos
        },
        "cube": {
            "id": cube,
            "pos": cube_pos
        },
        "cube2": {
            "id": cube2,
            "pos": cube2_pos
        },
        "cube3": {
            "id": cube3,
            "pos": cube3_pos
        },
        "cube4":{
            "id": target_sphere,
            "pos": target_sphere_pos
        }
        # "cube4":{
        #     "id":cube4,
        #     "pos":cube4_pos
        # }

    }

    return return_objects


def get_unit_vect(p1, p2):
    p1 = np.array([p1["x"], p1["z"]])
    p2 = np.array([p2["x"], p2["z"]])
    vect = p2 - p1
    vect = vect/math.sqrt(vect[0]*vect[0] + vect[1]*vect[1])
    return vect


def update_pos(objects):
    utils.objects_ready = False
    utils.object_data = None
    tdw.send_to_server({"$type": "get_objects_data"})
    while not utils.objects_ready:
        pass

    for obj in utils.object_data:
        if obj["id"] == objects["sphere"]["id"]:
            objects["sphere"]["pos"] = obj["position"]
        if obj["id"] == objects["cube"]["id"]:
            objects["cube"]["pos"] = obj["position"]
        if obj["id"] == objects["cube2"]["id"]:
            objects["cube2"]["pos"] = obj["position"]
        if obj["id"] == objects["cube3"]["id"]:
            objects["cube3"]["pos"] = obj["position"]
        if obj["id"] == objects["cube4"]["id"]:
            objects["cube4"]["pos"] = obj["position"]

    return objects


def demonstration_sol(objects, dir_name):
    frame_count = []
    start_frame = 0
    for i in range(16):
        sphere_pos = objects["sphere"]["pos"]
        cubes = [objects["cube"]["pos"], objects["cube2"]["pos"], objects["cube3"]["pos"], objects["cube4"]["pos"]]
        vect = get_unit_vect(sphere_pos, cubes[i%4])
        t = {
            "start_frame": start_frame,
            "end_frame": start_frame + 152,
            "source_pos": sphere_pos,
            "target_pos": cubes[i % 3]
        }
        frame_count.append(t)
        force_mag = random.uniform(50, 80)
        tdw.send_to_server({"$type": "apply_force_to_object",
                            "force": {"x": vect[0]*force_mag,
                                      "y": 0, "z": vect[1]*force_mag},
                            "id": objects["sphere"]["id"]})
        step_one_frame(150)
        update_pos(objects)

        start_frame += 152

    with open(os.path.join("dist", dir_name, "data.pickle"), "wb") as fp:
        pickle.dump(frame_count, fp)