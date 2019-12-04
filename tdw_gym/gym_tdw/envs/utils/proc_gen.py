import numpy as np
import random
import os


"""
Dictionary
0: Empty
1: Obstacle
2: Main Sphere
3: Touch Sphere
4: Cone
5: Goal inside space 
6: Wall piece
7:
8: Center of goal with variable walls
9: Ramp Area
10: Lever area
11: Outside ramp center
12:
13:
14: Inside ramp center
15: Push Sphere
16:
17: Stack
18: Long Cube
19: Long cube center

"""

def put_side(table, side, i, j):
    if side == 'l':
        table[i-2:i+3, j-2] = 6
    if side == 'r':
        table[i-2:i+3, j+2] = 6
    if side == 't':
        table[i-2, j-2:j+3] = 6
    if side == 'b':
        table[i+2, j-2:j+3] = 6
    return table


def add_random_cube_stack_target(table, no_stacks):
    while no_stacks > 0:
        i = random.randint(1, 14)
        j = random.randint(1, 7)
        if table[i, j] == 0:
            table[i, j] = 17
            no_stacks -= 1


def add_random_rectangle(table, no_rectangles):
    while no_rectangles > 0:
        i = random.randint(1, 14)
        j = random.randint(1, 7)
        orientation = random.choice([0, 1])
        if orientation == 0:
            if table[i, j] == 0 and table[i, j-1] == 0 and table[i, j+1] == 0:
                table[i, j-1:j+2] = 18
                table[i, j] = 19
                no_rectangles -= 1
        else:
            if table[i, j] == 0 and table[i-1, j] == 0 and table[i+1, j] == 0:
                table[i-1:i+2, j] = 18
                table[i, j] = 19
                no_rectangles -= 1


def add_random_obstacles(no_cubes, table):
    region_density = {
        "region_1": 0.1,
        "region_2": 0.1,
        "region_3": 0.1,
        "region_4": 0.1,
    }
    while no_cubes > 0:

        i = random.randint(1, 14)
        j = random.randint(1, 7)
        if 0 <= i < 8 and 0 <= j < 4:
            region = "region_1"
        elif 0 <= i < 8 and 4 <= j < 9:
            region = "region_2"
        elif 8 <= i < 16 and 0 <= j < 4:
            region = "region_3"
        else:
            region = "region_4"
        # Check if neighboring cubes are there then increase probability
        # if table[i,j-1] == 1 or table[i-1,j]:
        #     cube_choice = np.random.choice([1, 0], p=[region_density[region] + 0.3, 1 - (region_density[region] + 0.3)])
        # else:
        #     cube_choice = np.random.choice([1, 0], p=[region_density[region], 1 - region_density[region]])
        cube_choice = np.random.choice([1, 0], p=[region_density[region], 1 - region_density[region]])
        if table[i,j] == 0:
            sides_clear = 0
            if table[i -1, j] != 1:
                sides_clear += 1
            if table[i+1, j] != 1:
                sides_clear += 1
            if table[i, j+1] != 1:
                sides_clear += 1
            if table[i, j + 1] != 1:
                sides_clear += 1
            if sides_clear == 4:
                table[i, j] = cube_choice
                no_cubes -= cube_choice
        if no_cubes == 0:
            break


def add_random_target_with_cones(table):
    while True:
        i = random.randint(2, 14)
        j = random.randint(2, 7)
        if table[i, j-1] == 0 and table[i, j+1] == 0 and table[i-1, j] == 0 and table[i+1, j] == 0:
            table[i, j] = 2
            table[i-1, j] = 4
            table[i+1, j] = 4
            table[i, j+1] = 4
            table[i, j-1] = 4
            break


def add_random_target_sphere(table, no_sphere, surrounding_cones=False, diff=2):
    while no_sphere > 0:
        i = random.randint(1, 14)
        j = random.randint(1, 7)
        if table[i, j] == 0:
            table[i, j] = 2
            if surrounding_cones:
                area_start_i = i - diff if i - diff >= 0 else 0
                area_end_i = i + diff if i + diff <= 15 else 15
                area_start_j = j - diff if j - diff >= 0 else 0
                area_end_j = j + diff if j + diff <= 15 else 15
                add_random_cone(table, random.randint(5,7), area_start_i, area_end_i, area_start_j, area_end_j)
            no_sphere -= 1


def add_random_push_sphere(table, no_sphere):
    while no_sphere > 0:
        i = random.randint(1, 14)
        j = random.randint(1, 7)
        if table[i, j] == 0:
            table[i, j] = 15
            no_sphere -= 1


def add_random_cone(table, no_cones, area_start_i=None, area_end_i=None, area_start_j=None, area_end_j=None):
    if area_start_i is not None and area_end_i is not None and area_start_j is not None and area_end_j is not None:
        while no_cones > 0:
            i = random.randint(area_start_i, area_end_i)
            j = random.randint(area_start_j, area_end_j)
            if table[i, j] == 0:
                table[i, j] = 4
                no_cones -= 1
    else:
        while no_cones > 0:
            i = random.randint(0, 15)
            j = random.randint(0, 8)
            if table[i, j] == 0:
                table[i, j] = 4
                no_cones -= 1


def add_main_sphere(table, i=None, j=None, surrounding_cones=False, diff=2):
    if not i and not j:
        while True:
            i = random.randint(1, 14)
            j = random.randint(1, 7)
            if table[i, j] == 0:
                table[i, j] = 3
                if surrounding_cones:
                    area_start_i = i - diff if i - diff >= 0 else 0
                    area_end_i = i + diff if i + diff <= 15 else 15
                    area_start_j = j - diff if j - diff >= 0 else 0
                    area_end_j = j + diff if j + diff <= 15 else 15
                    add_random_cone(table, random.randint(1, 3), area_start_i, area_end_i, area_start_j, area_end_j)
                return True
    else:
        if table[i, j] == 0:
            table[i, j] = 3
            return True
        else:
            return False


def add_random_goal(table, no_sides, cone=False, inside_cone=False):
    sides = ["t", "b", "r", "l"]
    added_sides = []
    while True:
        i = random.randint(3, 12)
        j = random.randint(3, 5)


        # Check all the five square if they are empty
        if i < 2 or j < 2:
            continue
        if not table[i-2:i+3, j-2:j+3].any():
            # Makesure there are no wall on any side. Fix for the render function bug
            neighbor_count = 0
            if i > 2:
                if not np.any(np.isin(table[i - 3, j - 2:j + 3], [6])):
                    neighbor_count += 1
            else:
                neighbor_count += 1
            if i < 13:
                if not np.any(np.isin(table[i + 3, j - 2:j + 3], [6])):
                    neighbor_count += 1
            else:
                neighbor_count += 1
            if j < 5:
                if not np.any(np.isin(table[i - 2:i + 3, j + 3], [6])):
                    neighbor_count += 1
            else:
                neighbor_count += 1
            if j > 2:
                if not np.any(np.isin(table[i - 2:i + 3, j - 3], [6])):
                    neighbor_count += 1
            else:
                neighbor_count += 1
            if neighbor_count < 4:
                continue
            # Add goal area
            table[i - 2:i + 3, j - 2:j + 3] = 5
            # Add goal center
            table[i, j] = 8
            # Add inside cone
            if inside_cone:
                while True:
                    _i = random.randint(i-1, i+1)
                    _j = random.randint(j-1, j+1)
                    if table[_i, _j] == 5:
                        table[_i, _j] = 4
                        break
            # Check if any of the neighbors are table walls
            while no_sides > 0:
                random.shuffle(sides)
                side = sides[0]
                del sides[0]
                no_sides -= 1
                added_sides.append(side)
                table = put_side(table, side, i, j)
            break
    if cone:
        # Get one of the unadded sides and add cone to it
        while True:
            selected_side = random.choice(["t", "b", "r", "l"])
            if selected_side in ["t", "b"] and selected_side not in added_sides:
                area_start_j = j - 2
                area_end_j = j + 2
                if "t" not in added_sides and selected_side == "t":
                    area_start_i = i - 4 if i - 4 >= 0 else 0
                    area_end_i = i - 2 if i - 2 >= 0 else 0
                    break
                elif "b" not in added_sides and selected_side == "b":
                    area_start_i = i + 2 if i + 2 <= 15 else 15
                    area_end_i = i + 4 if i + 4 <= 15 else 15
                    break
            else:
                area_start_i = i - 2
                area_end_i = i + 2
                if "l" not in added_sides and selected_side == "l":
                    area_start_j = j - 4 if j - 4 >= 0 else 0
                    area_end_j = j - 2 if j - 2 >= 0 else 0
                    break
                elif "r" not in added_sides and selected_side == "r":
                    area_start_j = j + 2 if j + 2 <= 15 else 15
                    area_end_j = j + 4 if j + 2 <= 15 else 15
                    break
        add_random_cone(table, random.randint(3,5), area_start_i, area_end_i, area_start_j, area_end_j)
    return table


def add_walled_target(table, no_sides):
    sides = ["t", "b", "r", "l"]
    # val_i = [14]
    # val_j = [7]
    while True:

        # i = random.choice([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14])
        # j = random.choice([1,3,4,5,7])
        i = random.randint(3, 12)
        j = random.randint(3, 5)
        # i = val_i.pop(0)
        # j = val_j.pop(0)
        # Find a 3x3 goal area
        top_wall = bottom_wall = left_wall = right_wall = False
        if not table[i - 1:i + 2, j - 1:j + 2].any():
            enable_render = []
            # Check walls
            if i == 1:
                top_wall = True
            elif i == 14:
                bottom_wall = True
            if j == 1:
                left_wall = True
            elif j == 7:
                right_wall = True
            # Check for obstacles
            wall_start_i = 0 if i - 2 < 0 else i - 2
            wall_end_i = 16 if i + 3 > 16 else i + 3
            wall_start_j = 0 if j - 2 < 0 else j - 2
            wall_end_j = 9 if j + 3 > 9 else j + 3
            if not left_wall:
                if not table[wall_start_i:wall_end_i, wall_start_j].any():
                    left_wall = True
                    enable_render.append("l")
            if not top_wall:
                if not table[wall_start_i, wall_start_j: wall_end_j].any():
                    top_wall = True
                    enable_render.append("t")
            if not bottom_wall:
                if not table[wall_end_i-1, wall_start_j: wall_end_j].any():
                    bottom_wall = True
                    enable_render.append("b")
            if not right_wall:
                if not table[wall_start_i:wall_end_i, wall_end_j-1].any():
                    right_wall = True
                    enable_render.append("r")
            if left_wall and top_wall and bottom_wall and right_wall:
                table[i - 1:i + 2, j - 1:j + 2] = 12
                table[i, j] = 13
                if "l" in enable_render:
                    table[wall_start_i:wall_end_i, wall_start_j] = 6
                if "t" in enable_render:
                    table[wall_start_i, wall_start_j: wall_end_j] = 6
                if "b" in enable_render:
                    table[wall_end_i-1, wall_start_j: wall_end_j] = 6
                if "r" in enable_render:
                    table[wall_start_i:wall_end_i, wall_end_j-1] = 6
                break
    return table


def add_walled_target_long(table, no_sides):
    sides = ["t", "b", "r", "l"]
    val_i = [10]
    val_j = [1]
    goal_bottom = 3
    while True:
        i = random.randint(1, 12)
        j = random.randint(1, 7)
        # i = val_i.pop(0)
        # j = val_j.pop(0)
        # Find a 3x3 goal area
        top_wall = bottom_wall = left_wall = right_wall = False
        if not table[i - 1:i + goal_bottom, j - 1:j + goal_bottom].any():
            enable_render = []
            # Check walls
            if i == 1:
                top_wall = True
            elif i == 16 - goal_bottom:
                bottom_wall = True
            if j == 1:
                left_wall = True
            elif j == 7:
                right_wall = True
            # Check for obstacles
            wall_start_i = 0 if i - 2 < 0 else i - 2
            wall_end_i = 16 if i + goal_bottom+1 > 16 else i + goal_bottom+1
            wall_start_j = 0 if j - 2 < 0 else j - 2
            wall_end_j = 9 if j + 3 > 9 else j + 3
            if not left_wall:
                if not table[wall_start_i:wall_end_i, wall_start_j].any():
                    left_wall = True
                    enable_render.append("l")
            if not top_wall:
                if not table[wall_start_i, wall_start_j: wall_end_j].any():
                    top_wall = True
                    enable_render.append("t")
            if not bottom_wall:
                if not table[wall_end_i-1, wall_start_j: wall_end_j].any():
                    bottom_wall = True
                    enable_render.append("b")
            if not right_wall:
                if not table[wall_start_i:wall_end_i, wall_end_j-1].any():
                    right_wall = True
                    enable_render.append("r")
            print(enable_render)
            if left_wall and top_wall and bottom_wall and right_wall:
                table[i - 1:i + goal_bottom, j - 1:j + goal_bottom] = 12
                table[i, j] = 13
                if "l" in enable_render:
                    table[wall_start_i:wall_end_i, wall_start_j] = 6
                if "t" in enable_render:
                    table[wall_start_i, wall_start_j: wall_end_j] = 6
                if "b" in enable_render:
                    table[wall_end_i-1, wall_start_j: wall_end_j] = 6
                if "r" in enable_render:
                    table[wall_start_i:wall_end_i, wall_end_j-1] = 6
                break
    return table


def display_table(table):
    for i in range(0, 16):
        if i == 0:
            print("".join(["___" for _ in range(10)]))
        row = "|"
        for j in range(0, 9):
            if table[i, j] == 1:
                row += "|-|"
            elif table[i, j] == 2:
                row += " O "
            elif table[i, j] == 3:
                row += " 0 "
            elif table[i, j] == 4:
                row += "/_\\"
            elif table[i, j] == 5 or table[i,j] == 12:
                row += " g "
            elif table[i, j] == 6:
                row += " W "
            elif table[i, j] == 9:
                row += " R "
            elif table[i, j] == 10:
                row += " L "
            elif table[i, j] == 11:
                row += " Rc"
            elif table[i, j] == 14:
                row += "iRc"
            elif table[i, j] == 15:
                row += " Q "
            elif table[i, j] == 17:
                row += "|||"
            elif table[i, j] == 18:
                row += " X "
            elif table[i, j] == 19:
                row += "xCx"
            else:
                row += "   "
        row += "|"
        print(row)
        if i == 15:
            print("".join(["---" for _ in range(10)]))


def add_ramp(table, lever=False, inside_ramp=False):
    found_center = False
    for i in range(16):
        for j in range(8):
            if table[i,j] == 13:
                found_center = True
                break
        if found_center:
            break
    if found_center:
        # Check for possible locations of ramp

        # Randomly pick an offset based on distance of goal center and border
        ramp_offset = random.randint(0, 5)
        ramp_length = 3
        goal_width = 2
        while True:
            print(ramp_offset)
            possible_ramp_locations = []
            # Check top
            if i - (ramp_length + goal_width + ramp_offset) > -1 and j - 1 > -1 and j + 2 < 10:
                ramp_start = i - (ramp_offset + goal_width + ramp_length)
                ramp_end = i - goal_width
                if not table[ramp_start:ramp_end, j-1:j+2].any():
                    possible_ramp_locations.append("t")
            # Check bottom
            if i + ramp_offset + goal_width + ramp_length + 1 < 17 and j - 1 > -1 and j + 2 < 10:
                ramp_start = i + goal_width + ramp_offset + 1
                ramp_end = i + ramp_offset + goal_width + ramp_length + 1
                if not table[ramp_start:ramp_end, j - 1:j + 2].any():
                    possible_ramp_locations.append("b")
            # Check right
            if i - 1 > -1 and i + 2 < 17 and j + 7 < 9:
                ramp_start = j + goal_width + 1 + ramp_offset
                ramp_end = j + ramp_offset + goal_width + ramp_length + 1
                if not table[i - 1:i + 2, ramp_start:ramp_end].any():
                    possible_ramp_locations.append("r")
            # Check left
            if i - 1 > -1 and i + 2 < 17 and j - 6 > -1:
                ramp_start = j - goal_width + ramp_offset + ramp_length
                ramp_end = j - ramp_offset + goal_width
                if not table[i - 1:i + 2, ramp_start:ramp_end].any():
                    possible_ramp_locations.append("l")
            if possible_ramp_locations:
                break
            else:
                ramp_offset = random.randint(0, 5)
        random.shuffle(possible_ramp_locations)
        wall_start_i = 0 if i - 2 < 0 else i - 2
        wall_end_i = 16 if i + 3 > 16 else i + 3
        wall_start_j = 0 if j - 2 < 0 else j - 2
        wall_end_j = 9 if j + 3 > 9 else j + 3
        if possible_ramp_locations[0] == "l":
            table[i - 1:i + 2, j - 5:j - 2] = 9
            table[i, j - 4] = 11
            if i-2 > 0 and i+3 < 16 and lever:
                table[wall_start_i:wall_end_i, wall_start_j] = 10
        elif possible_ramp_locations[0] == "r":
            table[i - 1:i + 2, j + 3:j + 6] = 9
            table[i, j + 4] = 11
            if i-2 > 0 and i+3 < 16 and lever:
                table[wall_start_i:wall_end_i, wall_end_j-1] = 10
        elif possible_ramp_locations[0] == "t":
            ramp_start = i - (ramp_offset + goal_width + ramp_length)
            ramp_end = i - (goal_width + ramp_offset)
            table[ramp_start:ramp_end, j - 1:j + 2] = 9
            table[ramp_start + 1, j] = 11
            if j - 2 > -1 and j + 3 < 10 and lever:
                table[wall_start_i, wall_start_j:wall_end_j] = 10
        elif possible_ramp_locations[0] == "b":
            ramp_start = i + goal_width + ramp_offset + 1
            ramp_end = i + ramp_offset + goal_width + ramp_length + 1
            table[ramp_start:ramp_end, j - 1:j + 2] = 9
            table[ramp_start +1, j] = 11
            if j - 2 > -1 and j + 3 < 10 and lever:
                table[wall_end_i-1, wall_start_j:wall_end_j] = 10
                # Add target sphere
        inside_ramp_loc = possible_ramp_locations[1] if len(possible_ramp_locations) > 1 else possible_ramp_locations[0]
        # Put inside ramp
        if inside_ramp:
            if inside_ramp_loc == "l":
                ramp_inside_i = random.choice([i-1, i+1])
                ramp_inside_j = j-1
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i, ramp_inside_j+1] = 9
                target_ball_i = i - 1 if ramp_inside_i == i + 1 else i + 1
                target_ball_j = random.choice([j - 1, j + 1])
                table[target_ball_i, target_ball_j] = 2
            elif inside_ramp_loc == "t":
                ramp_inside_i = i-1
                ramp_inside_j = random.choice([j-1, j+1])
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i+1, ramp_inside_j] = 9
                target_ball_i = random.choice([i - 1, i + 1])
                target_ball_j = j - 1 if ramp_inside_j == j + 1 else j + 1
                table[target_ball_i, target_ball_j] = 2

            elif inside_ramp_loc == "r":
                ramp_inside_i = random.choice([i - 1, i + 1])
                ramp_inside_j = j + 1
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i, ramp_inside_j - 1] = 9
                target_ball_i = i - 1 if ramp_inside_i == i + 1 else i + 1
                target_ball_j = random.choice([j - 1, j + 1])
                table[target_ball_i, target_ball_j] = 2

            elif inside_ramp_loc == "b":
                ramp_inside_i = i + 1
                ramp_inside_j = random.choice([j - 1, j + 1])
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i - 1, ramp_inside_j] = 9
                target_ball_i = i - 1
                target_ball_j = random.choice([j - 1, j + 1])
                table[target_ball_i, target_ball_j] = 2
        else:
            while True:
                target_i = random.randint(i - 1, i + 1)
                target_j = random.randint(j - 1, j + 1)
                if target_i != i and target_j != j:
                    table[target_i, target_j] = 2
                    break


def add_ramp_goal(table, lever=False, inside_ramp=False):
    found_center = False
    for i in range(16):
        for j in range(8):
            if table[i,j] == 8:
                found_center = True
                break
        if found_center:
            break
    entry_type = [10]
    if found_center:
        # Check for possible locations of ramp
        possible_ramp_locations = []
        # Check top
        if i - 5 > -1 and j - 1 > -1 and j + 2 < 10:
            if not table[i-5:i-2, j-1:j+2].any():
                # table[i - 6:i - 3, j - 1:j + 2] = 9
                possible_ramp_locations.append("t")
        # Check bottom
        if i + 6 < 17 and j -1 > -1 and j + 2 < 10:
            if not table[i + 3:i + 6, j - 1:j + 2].any():
                # table[i + 4:i + 7, j - 1:j + 2] = 9
                possible_ramp_locations.append("b")
        # Check right
        if i - 1 > -1 and i + 2 < 17 and j + 7 < 9:
            if not table[i - 1:i + 2, j + 4:j + 7].any():
                # table[i - 1:i + 2, j + 4:j + 7] = 9
                possible_ramp_locations.append("r")
        # Check left
        if i - 1 > -1 and i + 2 < 17 and j - 6 > -1:
            if not table[i - 1:i + 2, j - 6:j - 3].any():
                # table[i - 1:i + 2, j - 6:j - 3] = 9
                possible_ramp_locations.append("l")
        random.shuffle(possible_ramp_locations)
        wall_start_i = 0 if i - 2 < 0 else i - 2
        wall_end_i = 16 if i + 3 > 16 else i + 3
        wall_start_j = 0 if j - 2 < 0 else j - 2
        wall_end_j = 9 if j + 3 > 9 else j + 3

        if possible_ramp_locations[0] == "l":
            # print(table[i - 1:i + 2, j - 6:j - 3])
            table[i - 1:i + 2, j - 6:j - 3] = 9
            table[i, j - 5] = 11
            if i-2 > 0 and i+3 < 16 and lever:
                table[wall_start_i:wall_end_i, wall_start_j] = 10
        elif possible_ramp_locations[0] == "r":
            # print(table[i - 1:i + 2, j + 4:j + 7])
            table[i - 1:i + 2, j + 4:j + 7] = 9
            table[i, j + 5] = 11
            if i-2 > 0 and i+3 < 16 and lever:
                table[wall_start_i:wall_end_i, wall_end_j-1] = 10
        elif possible_ramp_locations[0] == "t":
            # print(table[i - 5:i - 2, j - 1:j + 2])
            table[i - 5:i - 2, j - 1:j + 2] = 9
            table[i - 4, j] = 11
            if j - 2 > -1 and j + 3 < 10 and lever:
                table[wall_start_i, wall_start_j:wall_end_j] = 10
        elif possible_ramp_locations[0] == "b":
            # print(table[i + 3:i + 6, j - 1:j + 2])
            table[i + 3:i + 6, j - 1:j + 2] = 9
            table[i + 4, j] = 11
            if j - 2 > -1 and j + 3 < 10 and lever:
                table[wall_end_i-1, wall_start_j:wall_end_j] = 10
                # Add target sphere
        inside_ramp_loc = possible_ramp_locations[1] if len(possible_ramp_locations) > 1 else possible_ramp_locations[0]
        # Put inside ramp
        if inside_ramp:
            if inside_ramp_loc == "l":
                ramp_inside_i = random.choice([i-1, i+1])
                ramp_inside_j = j-1
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i, ramp_inside_j+1] = 9


            elif inside_ramp_loc == "t":
                ramp_inside_i = i-1
                ramp_inside_j = random.choice([j-1, j+1])
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i+1, ramp_inside_j] = 9

            elif inside_ramp_loc == "r":
                ramp_inside_i = random.choice([i - 1, i + 1])
                ramp_inside_j = j + 1
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i, ramp_inside_j - 1] = 9

            elif inside_ramp_loc == "b":
                ramp_inside_i = i + 1
                ramp_inside_j = random.choice([j - 1, j + 1])
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i - 1, ramp_inside_j] = 9
        else:
            while True:
                target_i = random.randint(i - 1, i + 1)
                target_j = random.randint(j - 1, j + 1)
                if target_i != i and target_j != j:
                    table[target_i, target_j] = 2
                    break


def add_ramp_long(table, lever=False, inside_ramp=False, target_type=2):
    found_center = False
    goal_bottom = 3
    for i in range(16):
        for j in range(8):
            if table[i,j] == 13:
                found_center = True
                break
        if found_center:
            break
    entry_type = [10]
    if found_center:
        # Check for possible locations of ramp
        possible_ramp_locations = []
        # Check top
        if i - 7 > -1 and j - 1 > -1 and j + 2 < 10:
            if not table[i-7:i-4, j-1:j+2].any():
                # table[i - 6:i - 3, j - 1:j + 2] = 9
                possible_ramp_locations.append("t")
        # Check bottom
        if i + 7 < 17 and j - 1 > -1 and j + 2 < 10:
            if not table[i + goal_bottom + 2:i + goal_bottom + 5, j - 1:j + 2].any():
                # table[i + 4:i + 7, j - 1:j + 2] = 9
                possible_ramp_locations.append("b")
        # Check right
        if i - 1 > -1 and i + 2 < 17 and j + 7 < 9:
            if not table[i - 1:i + 2, j + 4:j + 7].any():
                # table[i - 1:i + 2, j + 4:j + 7] = 9
                possible_ramp_locations.append("r")
        # Check left
        if i - 1 > -1 and i + 2 < 17 and j - 6 > -1:
            if not table[i - 1:i + 2, j - 6:j - 3].any():
                # table[i - 1:i + 2, j - 6:j - 3] = 9
                possible_ramp_locations.append("l")
        random.shuffle(possible_ramp_locations)
        wall_start_i = 0 if i - 2 < 0 else i - 2
        wall_end_i = 16 if i + goal_bottom+1 > 16 else i + goal_bottom+1
        wall_start_j = 0 if j - 2 < 0 else j - 2
        wall_end_j = 9 if j + 3 > 9 else j + 3
        if possible_ramp_locations[0] == "l":
            table[i - 1:i + 2, j - 6:j - 3] = 9
            table[i, j - 5] = 11
            if i-2 > 0 and i+3 < 16 and lever:
                table[wall_start_i:wall_end_i, wall_start_j] = 10
        elif possible_ramp_locations[0] == "r":
            table[i - 1:i + 2, j + 4:j + 7] = 9
            table[i, j + 5] = 11
            if i-2 > 0 and i+3 < 16 and lever:
                table[wall_start_i:wall_end_i, wall_end_j-1] = 10
        elif possible_ramp_locations[0] == "t":
            table[i - 6:i - 3, j - 1:j + 2] = 9
            table[i - 5, j] = 11
            if j - 2 > -1 and j + 3 < 10 and lever:
                table[wall_start_i, wall_start_j:wall_end_j] = 10
        elif possible_ramp_locations[0] == "b":
            table[i + goal_bottom+2:i + goal_bottom+5, j - 1:j + 2] = 9
            table[i + 5, j] = 11
            if j - 2 > -1 and j + 3 < 10 and lever:
                table[wall_end_i-1, wall_start_j:wall_end_j] = 10
                # Add target sphere
        inside_ramp_loc = possible_ramp_locations[1] if len(possible_ramp_locations) > 1 else possible_ramp_locations[0]
        # Put inside ramp
        if inside_ramp:
            if inside_ramp_loc == "l":
                ramp_inside_i = random.choice([i-1, i+1])
                ramp_inside_j = j-1
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i, ramp_inside_j+1] = 9
                target_ball_i = i - 1 if ramp_inside_i == i + 1 else i + 1
                target_ball_j = random.choice([j - 1, j + 1])
                table[target_ball_i, target_ball_j] = target_type
            elif inside_ramp_loc == "t":
                ramp_inside_i = i-1
                ramp_inside_j = random.choice([j-1, j+1])
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i+1, ramp_inside_j] = 9
                target_ball_i = random.choice([i - 1, i + 1])
                target_ball_j = j - 1 if ramp_inside_j == j + 1 else j + 1
                table[target_ball_i, target_ball_j] = target_type

            elif inside_ramp_loc == "r":
                ramp_inside_i = random.choice([i - 1, i + 1])
                ramp_inside_j = j + 1
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i, ramp_inside_j - 1] = 9
                target_ball_i = i - 1 if ramp_inside_i == i + 1 else i + 1
                target_ball_j = random.choice([j - 1, j + 1])
                table[target_ball_i, target_ball_j] = target_type

            elif inside_ramp_loc == "b":
                ramp_inside_i = i + 1
                ramp_inside_j = random.choice([j - 1, j + 1])
                table[ramp_inside_i, ramp_inside_j] = 14
                table[ramp_inside_i - 1, ramp_inside_j] = 9
                target_ball_i = i - 1
                target_ball_j = random.choice([j - 1, j + 1])
                table[target_ball_i, target_ball_j] = target_type
        else:
            while True:
                target_i = random.randint(i - 1, i + 1)
                target_j = random.randint(j - 1, j + 1)
                if target_i != i and target_j != j:
                    table[target_i, target_j] = target_type
                    break


def save_puzzle(puzzle_data, task, difficulty):
    if not os.path.isdir(f"data/task_{task}"):
        os.mkdir(f"data/task_{task}")
    if not os.path.isdir(f"data/task_{task}/difficulty_{difficulty}"):
        os.mkdir(f"data/task_{task}/difficulty_{difficulty}")
    puzzle_list = list(os.listdir(f"data/task_{task}/difficulty_{difficulty}"))
    puzzle_list = [e for e in puzzle_list if "npy" in e]
    puzzle_number = len(puzzle_list) + 1
    filename = f"puzzle_no_{puzzle_number}"
    print(puzzle_number)
    np.save(f"data/task_{task}/difficulty_{difficulty}/{filename}", puzzle_data)


def save_image(scene_image, task, difficulty):
    if not os.path.isdir(f"data/task_{task}"):
        os.mkdir(f"data/task_{task}")
    if not os.path.isdir(f"data/task_{task}/difficulty_{difficulty}"):
        os.mkdir(f"data/task_{task}/difficulty_{difficulty}")
    image_list = list(os.listdir(f"data/task_{task}/difficulty_{difficulty}"))
    image_list = [e for e in image_list if "png" in e]
    image_number = len(image_list) + 1
    filename = f"image_no_{image_number}.png"
    scene_image.save(f"data/task_{task}/difficulty_{difficulty}/{filename}")


def create_puzzle(task, difficulty):
    arr = np.zeros([16, 9])
    if task == 1:
        no_cube = random.randint(3, 6)

        # Easy Task touch the ball among cube obstacles
        if difficulty == 1:
            no_cube = random.randint(5, 8)
            add_main_sphere(arr)
            add_random_target_sphere(arr, 1)
            add_random_obstacles(no_cube, arr)
        if difficulty == 2:
            no_cube = random.randint(5, 8)
            add_main_sphere(arr)
            if random.choice([0, 1]) == 0:
                add_random_target_sphere(arr, 1)
            else:
                add_random_cube_stack_target(arr, 1)
            if random.choice([0, 1]) == 0:
                if random.choice([0, 1]) == 0:
                    add_random_target_sphere(arr, 1)
                else:
                    add_random_cube_stack_target(arr, 1)
            add_random_target_sphere(arr, 1)
            add_random_obstacles(no_cube, arr)


        # Increase difficulty by adding avoiding cone 1
        if difficulty == 3:
            add_main_sphere(arr, surrounding_cones=True)
            add_random_target_sphere(arr, 3)
            add_random_obstacles(no_cube, arr)

        # Yellow ball surrounded by cones
        if difficulty == 4:
            add_main_sphere(arr)
            add_random_target_with_cones(table=arr)
            add_random_rectangle(arr, 1)
            add_random_target_sphere(arr, 2)
            add_random_obstacles(4, arr)

        # Walled Target, Inside ramp,
        if difficulty == 5:
            add_walled_target(arr, 4)
            add_ramp(arr, inside_ramp=True)
            add_main_sphere(arr)
            add_random_target_sphere(arr, 2)
            add_random_obstacles(no_cube, arr)

        # Walled Target, No inside ramp, and avoidance cone :: Enforce order by touching walled target last
        if difficulty == 6:
            add_walled_target(arr, 4)
            add_ramp(arr)
            add_random_cone(arr, 1)
            add_main_sphere(arr)
            add_random_target_sphere(arr, 2)
            add_random_obstacles(no_cube, arr)

        # Add stacked target
        if difficulty == 7:
            add_random_cube_stack_target(arr, 1)
            add_main_sphere(arr)
            add_random_target_sphere(arr, 2)
            add_random_obstacles(no_cube, arr)

    elif task == 2:
        no_cube = random.randint(3, 5)

        # 3 Walled goal and push spheres 1-3
        if difficulty == 1:
            add_random_goal(arr, 3)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            add_random_push_sphere(arr, 1)
        if difficulty == 2:
            add_random_goal(arr, 3)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            add_random_push_sphere(arr, random.randint(2, 3))


        # 2 Walled goal and 1-3 push spheres
        if difficulty == 3:
            add_random_goal(arr, 2)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            if random.choice([0, 1]) == 0:
                add_random_rectangle(arr, 1)
            add_random_push_sphere(arr, 1)
        if difficulty == 4:
            add_random_goal(arr, 2)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            if random.choice([0, 1]) == 0:
                add_random_rectangle(arr, 1)
            add_random_push_sphere(arr, random.randint(2, 3))


        # 1 Walled goal and 1-3 push spheres
        if difficulty == 5:
            add_random_goal(arr, 1)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            add_random_rectangle(arr, 2)
            add_random_push_sphere(arr, 1)
        if difficulty == 6:
            add_random_goal(arr, 1)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            add_random_rectangle(arr, 2)
            add_random_push_sphere(arr, random.randint(2, 3))

        # Goal with No wall
        if difficulty == 7:
            add_random_goal(arr, no_sides=0)
            add_random_obstacles(no_cube, arr)
            add_random_rectangle(arr, 2)
            add_main_sphere(arr)
            add_random_push_sphere(arr, random.randint(2, 3))

        no_cube = random.randint(2, 3)
        # Goal with cone in it and 3 walls
        if difficulty == 8:
            add_random_goal(arr, no_sides=3, inside_cone=True)
            add_random_obstacles(no_cube, arr)
            add_random_rectangle(arr, 1)
            add_main_sphere(arr)
            add_random_push_sphere(arr, random.randint(2, 3))

        # Increase difficulty by putting a 4 sided goal and only ramp can be used to put the ball
        if difficulty == 9:
            add_random_goal(arr, no_sides=4)
            add_ramp_goal(arr, inside_ramp=True)
            add_random_obstacles(1, arr)
            add_random_rectangle(arr, 2)
            add_main_sphere(arr)
            add_random_push_sphere(arr, random.randint(1, 2))

    elif task == 3:
        # Simple task of touching and pushing
        no_cube = random.randint(2, 3)
        if difficulty == 1:
            add_random_goal(arr, 3)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            add_random_push_sphere(arr, 1)
            add_random_target_sphere(arr, 2)
        if difficulty == 2:
            add_random_goal(arr, 3)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            add_random_push_sphere(arr, 2)
            add_random_target_sphere(arr, 1)

        # Add avoidance cone to simple task
        if difficulty == 3:
            add_random_goal(arr, 4)
            add_ramp_goal(arr, inside_ramp=True)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            add_random_push_sphere(arr, 1)
            add_random_target_sphere(arr, 1)
            add_random_cone(arr, 1)
        if difficulty == 4:
            add_random_goal(arr, 4)
            add_ramp_goal(arr, inside_ramp=True)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            add_random_push_sphere(arr, 2)
            add_random_target_sphere(arr, 1)
            add_random_cone(arr, 1)

        # Add goal with avoidance cones
        if difficulty == 5:
            add_random_goal(arr, no_sides=3, cone=True)
            add_ramp_goal(arr)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            add_random_push_sphere(arr, 1)
            add_random_target_sphere(arr, 1)
        if difficulty == 6:
            add_random_goal(arr, no_sides=3, cone=True)
            add_ramp_goal(arr)
            add_random_obstacles(no_cube, arr)
            add_main_sphere(arr)
            add_random_push_sphere(arr, 2)
            add_random_target_sphere(arr, 1)

    return arr

def create_puzzle_poet_TASK1( is_main_sc, no_target, no_cube_stack_target, no_cones_target, no_walled_target, no_cubes, no_rectangles, is_ramp_inside ):
    # is_main_sc: if the main sphere surrounded by cones
    # no_target: # of target sphere
    # no_cube_stack_target: # of cube stack target
    # no_cones_target: # of cones surrounded targets
    # no_walled_target: # of walled target

    # no_cube: # of cubes
    # no_rectangles: # of rectangles
    # is_ramp_inside: if there's a ramp inside

    if is_ramp_inside:
        assert no_walled_target > 0

    arr = np.zeros([16, 9])
   
    add_main_sphere( arr, surrounding_cones=is_main_sc )
    if no_target > 0:
        add_random_target_sphere( arr, no_target )
    if no_cube_stack_target > 0:
        add_random_cube_stack_target( arr, no_cube_stack_target )
    for _ in range(no_cones_target):
        add_random_target_with_cones( arr )
    if no_walled_target > 0:
        add_walled_target( arr, no_walled_target )
        if is_ramp_inside:
            add_ramp( arr, inside_ramp=is_ramp_inside )

    if no_rectangles > 0:
        add_random_rectangle( arr, no_rectangles )
    if no_cubes > 0:
        add_random_obstacles( no_cubes, arr )

    return arr

def main():
    for _ in range(1):
        arr = create_puzzle(2, 24)
        display_table(arr)
    # np.save(f'data/cube_{no_cube}/puzzle_{9}.npy', arr)


if __name__ == '__main__':
    main()
