import numpy as np

#     is_main_sc=False,
    # no_target=1,
    # no_cube_stack_target=0,
    # no_cones_target=0,
    # no_walled_target=0,
    # no_cube=0,
    # no_rectangles=0, 
    # no_cones=0,
    # is_ramp_inside=False)


def env2array(env):
    arr = [0, 0, 0, 0, 0, 0]
    if env.is_main_sc:
        arr[0] = 1
    if env.is_ramp_inside:
        arr[1] = 1
    arr[2] = env.no_target
    arr[3] = env.no_cube_stack_target
    arr[4] = env.no_walled_target
    arr[5] = env.no_cube

    return arr

def euclidean_distance(nx, ny, normalize=False):

    x = np.array(env2array(nx))
    y = np.array(env2array(ny))

    x = x.astype(float)
    y = y.astype(float)

    # if normalize:
    #     norm = np.array([8., 8., 8., 3., 3.])
    #     x /= norm
    #     y /= norm

    # n, m = len(x), len(y)
    # if n > m:
    #     a = np.linalg.norm(y - x[:m])
    #     b = np.linalg.norm(y[-1] - x[m:])
    # else:
    #     a = np.linalg.norm(x - y[:n])
    #     b = np.linalg.norm(x[-1] - y[n:])

    assert len(x) == len(y) # why do i need to compare??
    a = np.linalg.norm( x - y )
    return np.sqrt(a**2)

def compute_novelty_vs_archive(archive, niche, k):
    # the sum of distances among K nearest neighbors
    distances = []
    normalize = False
    for point in archive.values():
        distances.append(euclidean_distance(point, niche, normalize=normalize)) # maybe you want to normalize, but it's turned off

    # Pick k nearest neighbors
    distances = np.array(distances)
    top_k_indicies = (distances).argsort()[:k]
    top_k = distances[top_k_indicies]
    return top_k.mean()
