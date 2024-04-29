import numpy as np

def maximise_distances(clew):
    """
    Maximise the distances between the worms.
    """
    total_cost = 0
    scaling_constant = 1000

    for i, worm in enumerate(clew):
        worm_cost = 0
        for j, other_worm in enumerate(clew):
            if i != j:
                dist = np.hypot(worm.x - other_worm.x, worm.y - other_worm.y)
                scaled_dist = dist * worm.r
                worm_cost = max(worm_cost, scaling_constant / (scaled_dist + 1.0))
        total_cost += worm_cost

    return total_cost


def maximise_global_variance(clew, variance_map):
    """
    Maximise the global variance that the worms cover.
    """
    total_cost = 0
    scaling_constant = 10

    for worm in clew:
        variances = np.reshape([variance_map[x, y] for x, y in worm.worm_slices], -1)
        total_cost += scaling_constant / (np.sum(variances) + 1.0)

    return total_cost


def minimise_local_variance(clew, img_map):
    """
    Minimise the variance of the image area that the worms cover
    """
    total_cost = 0

    for worm in clew:
        img_vals = np.reshape([img_map[x, y] for x, y in worm.worm_slices], -1)
        if len(img_vals) > 0:
            total_cost += np.std(img_vals)

    return total_cost / 10