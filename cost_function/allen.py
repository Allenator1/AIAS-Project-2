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
                dists = np.linalg.norm(worm.control_points() - other_worm.control_points(), axis=1)
                min_dist = np.min(dists) * worm.r
                worm_cost = max(worm_cost, scaling_constant / min_dist)
        total_cost += worm_cost

    return total_cost