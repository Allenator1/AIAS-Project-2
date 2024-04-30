import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2

from differential_evolution.DE import DE
from cost import *
from cost_function.yi import *
from Camo_Worm import Camo_Worm
import util

POPULATION_SIZE = 10


def final_cost(worm, var_map, image):
    """
    Final cost function that combines the cost functions from Robbie and Allen.
    """
    variance_cost = maximise_global_variance(worm, var_map)
    # camo_cost = minimise_local_variance(worm, image)
    return variance_cost


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'images/original.png'
    image = np.array(Image.open(image_path).convert('L'))
    image = np.flipud(image)

    height, width = image.shape
    worm_bounds = (0, width, 0, height)
    bounds = Camo_Worm.generate_bounds(worm_bounds)
    initial_population = [Camo_Worm(bounds) for _ in range(POPULATION_SIZE)]

    gaussian_img = cv2.GaussianBlur(image, (5, 5), 0, borderType=cv2.BORDER_REPLICATE)
    median_img = cv2.medianBlur(image, 5)
    var_map = np.abs(cv2.Laplacian(gaussian_img, cv2.CV_64F, borderType=cv2.BORDER_REPLICATE))
    var_map = cv2.normalize(var_map, None, 0, 1, cv2.NORM_MINMAX)

    cost_fn = lambda x: final_cost(x, var_map, median_img)

    # Create an instance of the DE algorithm
    de = DE(
        objective_function=cost_fn,
        bounds=bounds,
        initial_population=initial_population,
        max_iter=1000,        
        F=0.5,
        CR=0.9
    )

    # Run the DE algorithm
    while de.generation < de.max_iter:
        de.iterate()
        best_cost = cost_fn(de.get_best())
        print(f"Generation {de.generation}: Best cost = {best_cost}")

    # Get the best worm
    best_worm = de.get_best()

    # Visualize the best worm from the final population
    drawing = util.Drawing(image)
    drawing.add_worms([best_worm])
    drawing.show()
