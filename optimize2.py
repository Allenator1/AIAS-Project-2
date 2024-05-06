import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
from differential_evolution.DE import DE
from cost_function.robbie import straighten_worm
from cost_function.allen import maximise_distances
from cost_function.yi import *
from Camo_Worm import Camo_Worm, Clew
import util
import cv2
import numpy as np
from matplotlib import pyplot as plt

NUM_WORMS =2
print("start")
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = 'images/original.png'
image = np.array(Image.open(image_path).convert('L'))
image = np.flipud(image)

existing_clew_matrix = np.zeros_like(image, dtype=int)   #clew shown in matrix
print(existing_clew_matrix)
# empty_clew = Clew()

def final_cost(clew, image):
    """
    Final cost function that combines the cost functions from Robbie and Allen.
    """
    grid_shape = (20, 20)
    grid_variances = gen_variance_grid(image, grid_shape)
    layer_boundary_cost=layer_boundary_cost2(clew,gradient_y,image)
    intersection_cost=clews_intersection_cost_while_merging(clew,existing_clew_matrix,image)
    return  layer_boundary_cost+intersection_cost

if __name__ == '__main__':
    gradient_y = gradient_image_y(image)

    for i in range(40):  # Note: You don't need to specify the range(0, 4) as it's the default behavior
        bounds = Clew.generate_bounds(NUM_WORMS, image.shape)
        inibounds = Clew.generate_inibounds(NUM_WORMS, image.shape)
        initial_population = [Clew(inibounds) for _ in range(5)]

        cost_fn = lambda x: final_cost(x, image)

        # Create an instance of the DE algorithm
        de = DE(
            objective_function=cost_fn,
            bounds=bounds,
            initial_population=initial_population,
            max_iter=50,        
            F=0.05,
            CR=0.5
        )

        while de.generation < de.max_iter:
            de.iterate()
            print(f"Generation {de.generation}: Best cost = {cost_fn(de.get_best())}")
        best_clew = de.get_best()
        if i == 0:
            current_clew = best_clew
        else:
            current_clew.worms.extend(best_clew.worms)
            print(i)

        for worm in best_clew.worms:
            existing_clew_matrix = add_worms_to_existing_clew(existing_clew_matrix, worm_matrix(worm, image))
            print(np.sum(existing_clew_matrix))
    drawing = util.Drawing(image)
    drawing.add_worms(current_clew.worms)
    drawing.show()