from tdw_distributed.niches.tdwniches import Env_config
import numpy as np


def name_env_config(is_main_sc,
                    no_target, no_cube_stack_target, no_cones_target, no_walled_target,
                    no_cube, no_rectangles,
                    is_ramp_inside):
    # remember that I've changed this
    env_name = 'T_' + str(no_target) + '_' + str(no_cube_stack_target) + '_' + str(no_cones_target) + '_' + str(
        no_walled_target)
    env_name += '.cb' + str(no_cube) + '.r' + str(no_rectangles)
    env_name += '.ms' + str(is_main_sc) + '.ri' + str(is_ramp_inside)
    return env_name


class Reproducer:
    def __init__(self, args):
        self.rs = np.random.RandomState(args.master_seed)
        # print("### REPRODUCER -> categories: ", list(args.envs))
        self.categories = list(args.envs)

    def pick(self, arr):
        # randomly pick one 
        return self.rs.choice(arr)

    def populate_array(self, arr, default_value,
                       interval=0, increment=0, enforce=False, max_value=[]):
        assert isinstance(arr, list)
        if len(arr) == 0 or enforce:
            arr = list(default_value)
        elif len(max_value) == 2:
            choices = []
            for change0 in [increment, 0.0, -increment]:
                arr0 = np.round(arr[0] + change0, 1)
                if arr0 > max_value[0] or arr0 < default_value[0]:
                    continue
                for change1 in [increment, 0.0, -increment]:
                    arr1 = np.round(arr[1] + change1, 1)
                    if arr1 > max_value[1] or arr1 < default_value[1]:
                        continue
                    if change0 == 0.0 and change1 == 0.0:
                        continue
                    if arr0 + interval > arr1:
                        continue

                    choices.append([arr0, arr1])

            num_choices = len(choices)
            if num_choices > 0:
                idx = self.rs.randint(num_choices)
                # print(choices)
                # print("we pick ", choices[idx])
                arr[0] = choices[idx][0]
                arr[1] = choices[idx][1]

        return arr

    def mutate(self, parent):
        # TODO: maybe improve it
        # mutate according to the parent env, randomly change some of the args

        validate = False
        # bool
        is_main_sc = parent.is_main_sc
        is_ramp_inside = parent.is_ramp_inside
        # int
        no_target = parent.no_target
        no_cube_stack_target = parent.no_cube_stack_target
        no_cones_target = parent.no_cones_target
        no_walled_target = parent.no_walled_target

        no_cube = parent.no_cube
        no_rectangles = parent.no_rectangles

        while not validate:
            # obstacles
            no_cube = np.round(no_cube + np.random.uniform(-1, 1))
            max_cube = 5
            if no_cube > max_cube:
                no_cube = max_cube
            if no_cube <= 0:
                no_cube = 0

            no_rectangles = np.round(no_rectangles + np.random.uniform(-1, 1))
            max_rectangles = 0
            if no_rectangles > max_rectangles:
                no_rectangles = max_rectangles
            if no_rectangles <= 0:
                no_rectangles = 0

            # targets

            if 'target' in self.categories:
                no_target = np.round(no_target + np.random.uniform(-2, 2))
                max_target = 4
                if no_target > max_target:
                    no_target = max_target
                if no_target <= 0:
                    no_target = 0

            if 'cube_stack_target' in self.categories:
                no_cube_stack_target = np.round(no_cube_stack_target + np.random.uniform(-2, 2))
                max_cube_stack_target = 3
                if no_cube_stack_target > max_cube_stack_target:
                    no_cube_stack_target = max_cube_stack_target
                if no_cube_stack_target <= 0:
                    no_cube_stack_target = 0

            if 'cones_target' in self.categories:
                no_cones_target = np.round(no_cones_target + np.random.uniform(-1, 1))
                max_cones_target = 0
                if no_cones_target > max_cones_target:
                    no_cones_target = max_cones_target
                if no_cones_target <= 0:
                    no_cones_target = 0

            if 'walled_target' in self.categories:
                no_walled_target = np.round(no_walled_target + np.random.uniform(-1, 1))
                max_walled_target = 1
                if no_walled_target > max_walled_target:
                    no_walled_target = max_walled_target
                if no_walled_target <= 0:
                    no_walled_target = 0
                    is_ramp_inside = False
                else:
                    # because the walled target is disabled, no inside ramp will appear
                    rand = np.random.uniform(-1, 1)
                    if rand > 0.8:
                        is_ramp_inside = True
                    else:
                        is_ramp_inside = False

            rand = np.random.uniform(-1, 1)
            if rand > 0.5:
                is_main_sc = True
            else:
                is_main_sc = False

            if (is_main_sc == parent.is_main_sc) and \
                    (no_target == parent.no_target) and \
                    (no_cube_stack_target == parent.no_cube_stack_target) and \
                    (no_cones_target == parent.no_cones_target) and \
                    (no_walled_target == parent.no_walled_target) and \
                    (no_cube == parent.no_cube) and \
                    (no_rectangles == parent.no_rectangles) and (is_ramp_inside == parent.is_ramp_inside):
                exact_same = True
            else:
                exact_same = False

            validate = True
            no_tar = no_target + no_cube_stack_target + no_cones_target + no_walled_target
            if no_tar == 0:
                validate = False
            if exact_same:
                validate = False

        # print("## REPRODUCER_OP -> env config:  T: ", no_target, no_cube_stack_target, no_cones_target, no_walled_target)
        # print("## REPRODUCER_OP -> env config:  O: ", no_cube, no_rectangles)

        child_name = name_env_config(is_main_sc,
                                     no_target,
                                     no_cube_stack_target,
                                     no_cones_target,
                                     no_walled_target,
                                     no_cube,
                                     no_rectangles,
                                     is_ramp_inside)
        # print("## REPRODUCER_OP -> env config:  NAME: ", child_name)

        # from IPython import embed
        # embed()

        child = Env_config(name=child_name,
                           is_main_sc=is_main_sc,
                           no_target=no_target,
                           no_cube_stack_target=no_cube_stack_target,
                           no_cones_target=no_cones_target,
                           no_walled_target=no_walled_target,
                           no_cube=no_cube,
                           no_rectangles=no_rectangles,
                           is_ramp_inside=False)
        return child
