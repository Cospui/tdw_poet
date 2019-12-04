import gym
from gym_tdw.envs.utils.proc_gen import create_puzzle, display_table, create_puzzle_poet_TASK1

from PIL import Image
import numpy as np

env = gym.make('gym_tdw:tdw_puzzle_proc-v0', tdw_ip='169.48.98.28', self_ip="162.133.74.249")
task = 1
difficulty = 1
env.set_observation(True)
puzzle_data = create_puzzle_poet_TASK1( is_main_sc=False,
                                        no_target=1,
                                        no_cube_stack_target=1,
                                        no_cones_target=0,
                                        no_walled_target=0,
                                        no_cubes=1,
                                        no_rectangles=0,
                                        is_ramp_inside=False)
display_table(puzzle_data)
env.add_change_puzzle(puzzle_data)

image_no = 0
action = {"x":0, "z":0}
obs, reward, episode_done, _ = env.step(action)
img = obs['image']
img = Image.fromarray(img.astype(np.uint8))
img.save('view'+str(1)+'.png')


obj_info = obs['object_information']
for value in obj_info.values():
    print( value['position'] )
    print( value['model_name'] )
    print( value['color_id'] )

print( obj_info )

env.close()