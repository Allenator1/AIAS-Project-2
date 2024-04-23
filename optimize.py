import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image

from differential_evolution.DE import DE
from cost_function.robbie import simplified_cost
from Camo_Worm import Camo_Worm, Clew
import util

NUM_WORMS = 10

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'images/denoised.png'
    image = np.array(Image.open(image_path).convert('L'))
    image = np.flipud(image)

    bounds = Clew.generate_bounds(NUM_WORMS, image.shape)
    initial_population = [Clew(bounds) for _ in range(NUM_WORMS)]

    cost_fn = lambda x: simplified_cost(x, image)

    # Create an instance of the DE algorithm
    de = DE(
        objective_function=cost_fn,
        bounds=bounds,
        initial_population=initial_population,
        max_iter=100,        
        F=0.5,
        CR=0.9
    )

    # Run the DE algorithm
    while de.generation < de.max_iter:
        de.iterate()
        print(f"Generation {de.generation}: Best cost = {simplified_cost(de.get_best(), image)}")

    # Get the best clew
    best_clew = de.get_best()

    # Visualize the best clew from the final population
    drawing = util.Drawing(image)
    drawing.add_worms(best_clew.worms)
    drawing.show()
