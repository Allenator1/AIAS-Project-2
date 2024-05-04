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

POPULATION_SIZE = 24


def final_cost(worm, var_map, image):
    """
    Final cost function that combines the cost functions from Robbie and Allen.
    """
    camo_cost = minimise_local_variance(worm, image)
    grad_cost = maximise_grad(worm, var_map)
    return camo_cost + grad_cost


def generate_optimal_worms(image, max_depth=4, max_iter=100, population_size=24, F=0.1, CR=0.5):
    worms = []
    im_height, im_width = image.shape

    median_img = cv2.medianBlur(image, 9)
    grad_y = np.abs(cv2.Sobel(median_img, cv2.CV_64F, 0, 1, ksize=5))
    # Rescale the gradient magnitude to lie between 0 and 1
    grad_y = (grad_y - np.min(grad_y)) / (np.max(grad_y) - np.min(grad_y))
    grad_y = (grad_y > 0.1).astype(np.float32)

    plt.imshow(grad_y, cmap='gray')

    def subdivision_worm(x, y, height, width, recursion_depth):
        worm_bounds = (x, x + width, y, y + height)
        bounds = Camo_Worm.generate_bounds(worm_bounds)
        initial_population = [Camo_Worm(bounds) for _ in range(population_size)]

        cost_fn = lambda wrm: final_cost(wrm, grad_y, median_img)

        de = DE(
            objective_function=cost_fn,
            bounds=bounds,
            initial_population=initial_population,
            max_iter=max_iter,        
            F=F,
            CR=CR
        )

        best_cost = np.inf
        while de.generation < de.max_iter:
            de.iterate()
            best_worm = de.get_best()
            best_cost = cost_fn(best_worm)
            # print(f"Generation {de.generation}: Best cost = {best_cost}")

        indent = "==" * (max_depth - recursion_depth)
        print(f"{indent} Depth = {max_depth - recursion_depth} Best cost = {best_cost}")
        if best_cost > 0.0:
            if recursion_depth > 0 and width > 2 and height > 2:
                new_height = height // 2
                new_width = width // 2
                subdivision_worm(x, y, new_height, new_width, recursion_depth - 1)
                subdivision_worm(x + new_width, y, new_height, new_width, recursion_depth - 1)
                subdivision_worm(x, y + new_height, new_height, new_width, recursion_depth - 1)
                subdivision_worm(x + new_width, y + new_height, new_height, new_width, recursion_depth - 1)
            else:
                worms.append(best_worm)
        else:
            worms.append(best_worm)

    subdivision_worm(0, 0, im_height, im_width, max_depth)
    return worms


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'images/original.png'
    image = np.array(Image.open(image_path).convert('L'))
    image = np.flipud(image)

    # Generate the optimal worms
    final_worms = generate_optimal_worms(image)

    # Visualize the best worm from the final population
    drawing = util.Drawing(image)
    drawing.add_worms(final_worms)
    drawing.show()
