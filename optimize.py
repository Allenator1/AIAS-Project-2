import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2

from differential_evolution.DE import DE
from cost_function.robbie import *
from cost_function.allen import *
from cost_function.yi import *
from Camo_Worm import Camo_Worm, Clew
import util

NUM_WORMS = 50


def final_cost(clew, var_map, image):
    """
    Final cost function that combines the cost functions from Robbie and Allen.
    """
    distance_cost = maximise_distances(clew)
    variance_cost = maximise_global_variance(clew, var_map)
    camo_cost = minimise_local_variance(clew, image)
    return  3 * camo_cost + 1 * variance_cost + 2 * distance_cost


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'images/original.png'
    image = np.array(Image.open(image_path).convert('L'))
    image = np.flipud(image)

    bounds = Clew.generate_bounds(NUM_WORMS, image.shape)
    initial_population = [Clew(bounds) for _ in range(NUM_WORMS)]

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0, borderType=cv2.BORDER_REPLICATE)
    median_filter = cv2.medianBlur(image, 5)
    var_map = cv2.Laplacian(blurred_image, cv2.CV_64F, borderType=cv2.BORDER_REPLICATE)
    var_map = np.abs(var_map)
    var_map = cv2.normalize(var_map, None, 0, 1, cv2.NORM_MINMAX)

    cost_fn = lambda x: final_cost(x, var_map, image)

    # Create an instance of the DE algorithm
    de = DE(
        objective_function=cost_fn,
        bounds=bounds,
        initial_population=initial_population,
        max_iter=100,        
        F=0.1,
        CR=0.9
    )

    # Run the DE algorithm
    while de.generation < de.max_iter:
        de.iterate()
        print(f"Generation {de.generation}: Best cost = {cost_fn(de.get_best())}")

    # Get the best clew
    best_clew = de.get_best()

    # Visualize the best clew from the final population
    drawing = util.Drawing(image)
    drawing.add_worms(best_clew.worms)
    drawing.show()
