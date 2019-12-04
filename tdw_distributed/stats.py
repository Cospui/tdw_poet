# The following code is from openai/evolution-strategies-starter
# (https://github.com/openai/evolution-strategies-starter)
# under the MIT License.


import numpy as np


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    """
    compute: SUM( Ei * noise_i )
    """
    # weights: proc_returns[:,0] - proc_returns[:,1] (after normalization)
    # vecs is noise
    total = 0.
    num_items_summed = 0
    
    idx_weight = 0
    for batch_vec in vecs:
        total += batch_vec * weights[idx_weight]
        idx_weight += 1

    # print( "######### STATS -> batch w sum : ", total , " @ " , idx_weight )
    return total, idx_weight
