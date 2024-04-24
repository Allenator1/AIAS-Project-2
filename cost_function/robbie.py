import numpy as np

def straighten_worm(clew):
    """
    Simplified cost function.
    """
    total_cost = 0
    desired_length = 100

    for worm in clew:
        # Get the intermediate points along the worm's path
        intermediate_points = worm.intermediate_points()

        # Calculate the deviation from a straight line
        straight_diff = abs(worm.approx_length() - np.linalg.norm(intermediate_points[-1] - intermediate_points[0]))
        straight_diff = np.exp(straight_diff) - 1
        
        # Calculate the length difference from the desired length
        # length_diff = abs(worm.approx_length() - desired_length)

        # Combine the length difference and deviation into a single cost value
        total_cost += straight_diff

    return total_cost