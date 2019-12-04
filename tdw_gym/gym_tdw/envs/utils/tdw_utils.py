import utils.object_utils as object_utils
import random



def create_obstacles():
    number_of_cubes = random.randint()
    for i  in range(number_of_cubes):
        positions = generate_co_rdinates()

def generate_co_rdinates():
    x_bounds = [-4.67, -3.584]
    z_bounds = [-5.656, -3.571]
    position = {"x": random.uniform(x_bounds[0], x_bounds[1]),
                "y": 0.8749,
                "z": random.uniform(z_bounds[0], z_bounds[1])
                }
