import pygame
import gym
from gym_tdw.envs.utils.proc_gen import create_puzzle, display_table, save_puzzle, save_image
import numpy as np
from PIL import Image

pygame.init()


def _create_puzzle(task, difficulty):
    while True:
        try:
            puzzle_data = create_puzzle(task, difficulty)
            return puzzle_data
        except:
            continue


def env_create_puzzle(ienv, itask, idifficulty):
    ipuzzle_data = _create_puzzle(itask, idifficulty)
    display_table(ipuzzle_data)
    env.add_change_puzzle(ipuzzle_data)
    action_ = {
        "x": 0,
        "z": 0,
        "stop": False
    }
    env.set_observation(True)
    iobs, _, _, _ = ienv.step(action_)
    env.set_observation(False)
    iimg = iobs["image"]
    return Image.fromarray(iimg), ipuzzle_data

# env = gym.make('gym_tdw:tdw_puzzle-v0', tdw_ip='localhost', self_ip=None)
# env.add_change_puzzle(5)
env = gym.make('gym_tdw:tdw_puzzle_proc-v0', tdw_ip='localhost', self_ip=None, debug=False)
task = 1
difficulty = 1
_, puzzle_data = env_create_puzzle(env, task, difficulty)
pressed_left = pressed_right = pressed_up = pressed_down = power_up = stop = False
force_mag = 1
_task = None
_difficulty_0 = None
_difficulty_1 = None
_difficulty = None
scene_image = None
while True:
    action = {
        "x": 0,
        "z": 0
    }
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:  # check for key presses
            if event.key == pygame.K_LEFT or event.key == pygame.K_a:  # left arrow turns left
                pressed_left = True
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:  # right arrow turns right
                pressed_right = True
            elif event.key == pygame.K_UP or event.key == pygame.K_w:  # up arrow goes up
                pressed_up = True
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:  # down arrow goes down
                pressed_down = True
            elif event.key == pygame.K_z or event.key == pygame.K_m:
                power_up = True
            elif event.key == pygame.K_r:
                env.reset()
            elif event.key == pygame.K_c:
                scene_image, puzzle_data = env_create_puzzle(env, task, difficulty)
            elif event.key == pygame.K_x or event.key == pygame.K_n:
                stop = True
            elif event.key == pygame.K_u:
                print("Saving scene")
                save_puzzle(puzzle_data, task, difficulty)
                save_image(scene_image, task, difficulty)
            elif event.key == pygame.K_1:
                if _task is None:
                    _task = 1
                elif _difficulty_0 is None:
                    _difficulty_0 = 1
                elif _difficulty_1 is None:
                    _difficulty_1 = 1
            elif event.key == pygame.K_2:
                if _task is None:
                    _task = 2
                elif _difficulty_0 is None:
                    _difficulty_0 = 2
                elif _difficulty_1 is None:
                    _difficulty_1 = 2
            elif event.key == pygame.K_3:
                if _task is None:
                    _task = 3
                elif _difficulty_0 is None:
                    _difficulty_0 = 3
                elif _difficulty_1 is None:
                    _difficulty_1 = 3
            elif event.key == pygame.K_4:
                if _task is None:
                    _task = 4
                elif _difficulty_0 is None:
                    _difficulty_0 = 4
                elif _difficulty_1 is None:
                    _difficulty_1 = 4
            elif event.key == pygame.K_5:
                if _task is None:
                    _task = 5
                elif _difficulty_0 is None:
                    _difficulty_0 = 5
                elif _difficulty_1 is None:
                    _difficulty_1 = 5
            elif event.key == pygame.K_6:
                if _task is None:
                    _task = 6
                elif _difficulty_0 is None:
                    _difficulty_0 = 6
                elif _difficulty_1 is None:
                    _difficulty_1 = 6
            elif event.key == pygame.K_7:
                if _task is None:
                    _task = 7
                elif _difficulty_0 is None:
                    _difficulty_0 = 7
                elif _difficulty_1 is None:
                    _difficulty_1 = 7
            elif event.key == pygame.K_8:
                if _task is None:
                    _task = 8
                elif _difficulty_0 is None:
                    _difficulty_0 = 8
                elif _difficulty_1 is None:
                    _difficulty_1 = 8
            elif event.key == pygame.K_9:
                if _task is None:
                    _task = 9
                elif _difficulty_0 is None:
                    _difficulty_0 = 9
                elif _difficulty_1 is None:
                    _difficulty_1 = 9
            elif event.key == pygame.K_0:
                if _task is None:
                    _task = 0
                elif _difficulty_0 is None:
                    _difficulty_0 = 0
                elif _difficulty_1 is None:
                    _difficulty_1 = 0

        elif event.type == pygame.KEYUP:  # check for key releases
            if event.key == pygame.K_LEFT or event.key == pygame.K_a:  # left arrow turns left
                pressed_left = False
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:  # right arrow turns right
                pressed_right = False
            elif event.key == pygame.K_UP or event.key == pygame.K_w:  # up arrow goes up
                pressed_up = False
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:  # down arrow goes down
                pressed_down = False
            elif event.key == pygame.K_z or event.key == pygame.K_m:
                power_up = False
            elif event.key == pygame.K_x or event.key == pygame.K_n:
                stop = False
    if _task is not None and _difficulty_0 is not None and _difficulty_1 is not None:
        _difficulty = _difficulty_0 * 10 + _difficulty_1
        if 1 <= _task <= 3:
            if _task == 1:
                if not 1 <= _difficulty <= 8:
                    _task, _difficulty_0, _difficulty_1, _difficulty = None, None, None, None
            elif _task == 2:
                if not 1 <= _difficulty <= 13:
                    _task, _difficulty_0, _difficulty_1, _difficulty = None, None, None, None
            elif _task == 3:
                if not 1 <= _difficulty <= 6:
                    _task, _difficulty_0, _difficulty_1, _difficulty = None, None, None, None
        else:
            _task, _difficulty_0, _difficulty_1, _difficulty = None, None, None, None
    if _task and _difficulty:
        task = _task
        difficulty = _difficulty
        scene_image, puzzle_data = env_create_puzzle(env, task, difficulty)
        _task = None
        _difficulty_0 = None
        _difficulty_1 = None
        _difficulty = None
    power_up_force = 100 if power_up else 0
    action = {
        "x": 0,
        "z": 0,
        "stop": stop
    }
    if pressed_left:
        # print("Left key down")
        action["x"] += -1*(force_mag + power_up)

    if pressed_right:
        # print("Right key down")
        action["x"] += 1 * (force_mag + power_up)

    if pressed_up:
        # print("Up key down")
        action["z"] += 1 * (force_mag + power_up)

    if pressed_down:
        # print("Down key down")
        action["z"] += -1 * (force_mag + power_up)
    # print(f" Task {_task} diff_0 {_difficulty_0} diff_1 {_difficulty_1}")
    obs, reward, episode_done, _ = env.step(action)



