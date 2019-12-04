from skimage.draw import line, polygon
from skimage import io
import os
import numpy as np

def post_processing_arrow(dir_name, frame_count):

    for event in frame_count:
        start_frame = event["start_frame"]
        end_frame = event["end_frame"]
        source_pos = [event["source_pos"]["x"], event["source_pos"]["z"]]
        target_pos = [event["target_pos"]["x"], event["target_pos"]["z"]]

        # Read images and process each one of them
        source_pos = translate_pos(source_pos)
        target_pos = translate_pos(target_pos)
        _ = [process_image(dir_name, i, source_pos, target_pos) for i in range(start_frame, end_frame)]



def process_image(dir_name, i, source_pos, target_pos):
    add_arrow(source_pos, target_pos, os.path.join(dir_name, "image-{0}.png".format(i)), os.path.join("dist", "demo_1", "post", "image-{0}.png".format(i)))
    return 1


def translate_pos(v):
    v[0] = round(abs((v[0] + 5.823)/ 0.0051))
    v[1] = round(abs((v[1] + 3.386)/ 0.0051))
    v_ = [v[1], v[0]]
    return v_


def move_point_back(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = abs(dx) + abs(dy)
    dx = dx / norm
    dy = dy / norm
    mag = 20
    start = [round(start[0] - dx * mag), round(start[1] - dy * mag)]
    mag = 60
    end = [round(start[0] - dx * mag), round(start[1] - dy * mag)]
    while end[0] >= 480 or end[1] >= 960:
        mag -= 1
        end = [round(start[0] - dx * mag), round(start[1] - dy * mag)]
    return start, end


def add_arrow(start, end, img_path, img_save_path):

    start, end = move_point_back(start, end)
    img = io.imread(img_path)
    # Draw arrow line
    rr, cc = line(start[0], start[1], end[0], end[1])
    img[rr, cc, 0] = 255
    img[rr, cc, 1] = 255
    img[rr, cc, 2] = 0

    # start = [start[0] + 1, start[1] + 1]
    # end = [end[0] + 1, end[1] + 1]
    # dx = end[0] - start[0]
    # dy = end[1] - start[1]
    # rr, cc = line(start[0], start[1], start[0] - dx, start[1] - dy)
    # img[rr, cc, 0] = 255
    # img[rr, cc, 1] = 255
    # img[rr, cc, 2] = 0

    # # Draw arrow tip
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = abs(dx) + abs(dy)
    dx = dx/norm
    dy = dy/norm
    if dy == 0:
        if start[0] > end[0]:
            r = np.array([start[0], start[0], start[0] + 7])
        else:
            r = np.array([start[0], start[0], start[0] - 7])
        c = np.array(
            [start[1] - 5, start[1] + 5, start[1]])
    else:
        perpendicular_slope = -1 * dx / dy
        y_intercept = start[1] - perpendicular_slope * start[0]
        x_1 = start[0] - 7*dy
        x_2 = start[0] + 7*dy
        r = np.array([x_1, x_2, start[0] - 8*dx])
        c = np.array([x_1*perpendicular_slope + y_intercept, x_2*perpendicular_slope + y_intercept, start[1] - 8*dy])
    rr, cc = polygon(r, c)
    img[rr, cc, 0] = 255
    img[rr, cc, 1] = 255
    img[rr, cc, 2] = 0
    io.imsave(img_save_path, img)