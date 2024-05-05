import numpy as np


def maximise_grad(worm, variance_map):
    """
    Maximise the global variance that the worms cover.
    """
    vars = variance_map[worm.indices]
    return -np.sum(vars)


def minimise_local_variance(worm, img_map):
    """
    Minimise the variance of the image area that the worms cover
    """
    img_vals = img_map[worm.indices]
    camo_cost = np.std(img_vals) * len(img_vals)
    return camo_cost


def straighten_worm(worm):
    """
    Calculate the cost of a clew of worms based on the deviation from a straight line.
    """
    # Get the intermediate points along the worm's path
    intermediate_points = worm.intermediate_points()

    # Calculate the deviation from a straight line
    straight_diff = abs(worm.approx_length() - np.linalg.norm(intermediate_points[-1] - intermediate_points[0]))
    straight_diff = np.exp(straight_diff) - 1

    # Combine the length difference and deviation into a single cost value
    return straight_diff