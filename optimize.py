import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

from differential_evolution.DE import DE
from cost import *
from cost_function.yi import *
from Camo_Worm import Camo_Worm
from anchor import generate_boxes
from tqdm import tqdm
import util

POPULATION_SIZE = 24
MAX_ITER = 100
F = 0.5
CR = 0.9


def final_cost(worm, var_map, image, worm_mask=None):
    """
    Final cost function that combines the cost functions from Robbie and Allen.
    """
    camo_cost = minimise_local_variance(worm, image)
    grad_cost = maximise_grad(worm, var_map)
    overlap_cost = 0
    if worm_mask is not None:
        overlap_cost = minimise_overlap(worm, worm_mask)
    return 40 * grad_cost + camo_cost + overlap_cost - worm.length


def optimise_worm(x1, y1, width, height, grad_y, median_img, worm_mask=None):
    worm_bounds = (x1, x1 + width, y1, y1 + height)
    bounds = Camo_Worm.generate_bounds(worm_bounds)
    initial_population = [Camo_Worm(bounds) for _ in range(POPULATION_SIZE)]

    cost_fn = lambda wrm: final_cost(wrm, grad_y, median_img, worm_mask)

    de = DE(
        objective_function=cost_fn,
        bounds=bounds,
        initial_population=initial_population,
        max_iter=MAX_ITER,        
        F=F,
        CR=CR
    )

    while de.generation < de.max_iter:
        de.iterate()
        best_worm = de.get_best()
        best_cost = cost_fn(best_worm) / (width * height)

    return (best_worm, best_cost)


def recursive_subdivision_optimisation(image, max_depth=4):
    worms = []
    im_height, im_width = image.shape

    median_img = cv2.medianBlur(image, 11)
    grad_y = np.abs(cv2.Sobel(median_img, cv2.CV_64F, 0, 1, ksize=5))
    # Rescale the gradient magnitude to lie between 0 and 1
    grad_y = (grad_y - np.min(grad_y)) / (np.max(grad_y) - np.min(grad_y))
    grad_y = (grad_y > 0.1).astype(np.float32)

    def subdivision_worm(x, y, height, width, recursion_depth):
        best_worm, best_cost = optimise_worm(x, y, width, height, grad_y, median_img)
        indent = "==" * (max_depth - recursion_depth)
        print(f"{indent} Depth = {max_depth - recursion_depth} Best cost = {best_cost}")

        new_height = height // 2
        new_width = width // 2

        if recursion_depth > 0 and width > 2 and height > 2:
            subdivision_worm(x, y, new_height, new_width, recursion_depth - 1)
            subdivision_worm(x + new_width, y, new_height, new_width, recursion_depth - 1)
            subdivision_worm(x, y + new_height, new_height, new_width, recursion_depth - 1)
            subdivision_worm(x + new_width, y + new_height, new_height, new_width, recursion_depth - 1)
            
        worms.append(best_worm)

    subdivision_worm(0, 0, im_height, im_width, max_depth)
    return worms


def multiscale_optimisation(image):
    worms = []

    median_img = cv2.medianBlur(image, 11)
    grad_y = np.abs(cv2.Sobel(median_img, cv2.CV_64F, 0, 1, ksize=5))
    # Rescale the gradient magnitude to lie between 0 and 1
    grad_y = (grad_y - np.min(grad_y)) / (np.max(grad_y) - np.min(grad_y))
    grad_y = (grad_y > 0.1).astype(np.float32)

    for y1, x1, y2, x2 in tqdm(generate_boxes(image.shape)):
        best_worm, _ = optimise_worm(x1, y1, x2 - x1, y2 - y1, grad_y, median_img)
        worms.append(best_worm)
        
    return worms


def iterative_optimisation(image, num_iter=100):
    worms = []
    worm_mask = np.zeros(image.shape, dtype=np.uint8)

    median_img = cv2.medianBlur(image, 3)
    grad_y = np.abs(cv2.Sobel(median_img, cv2.CV_64F, 0, 1, ksize=3))
    # Rescale the gradient magnitude to lie between 0 and 1
    grad_y = (grad_y - np.min(grad_y)) / (np.max(grad_y) - np.min(grad_y))
    grad_y = (grad_y > 0.10).astype(np.float32)

    for _ in tqdm(range(num_iter)):
        best_worm, _ = optimise_worm(0, 0, image.shape[1], image.shape[0], grad_y, median_img, worm_mask)

        worms.append(best_worm)
        new_mask = np.zeros(image.shape, dtype=np.uint8)
        new_mask[best_worm.indices] = 1
        new_mask = cv2.dilate(new_mask, np.ones((best_worm.width + 4, best_worm.width + 4), np.uint8), iterations=1)
        worm_mask = np.logical_or(worm_mask, new_mask).astype(np.uint8)

    return worms


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'images/original.png'
    mask = [320, 560, 160, 880]  # ymin ymax xmin xmax

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    image = np.flipud(image)[50:height - 50, 50:width - 50]

    # Generate the optimal worms
    final_worms = iterative_optimisation(image)

    # Visualize the best worm from the final population
    drawing = util.Drawing(image)
    drawing.add_worms(final_worms)
    drawing.show()
