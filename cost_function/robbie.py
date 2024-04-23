import numpy as np

def simplified_cost(clew, image):
    """
    Simplified cost function.
    """
    total_cost = 0

    for worm in clew:
        # Get the intermediate points along the worm's path
        intermediate_points = worm.intermediate_points()

        # Calculate the difference between the worm's length and the desired length (e.g., 100 pixels)
        length_diff = np.abs(worm.approx_length() - 100)

        # Calculate the deviation from a straight line
        deviation = np.linalg.norm(intermediate_points[-1] - intermediate_points[0])

        # Combine the length difference and deviation into a single cost value
        total_cost += length_diff + 0.1 * deviation  # Adjust the weight as needed

    return total_cost