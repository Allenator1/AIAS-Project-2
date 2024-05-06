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
import numpy as np
from matplotlib import pyplot as plt

NUM_WORMS =1                               #add n worm at a time
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

    high_variance_cost=more_worms_to_high_variance_area(clew, grid_variances,image,grid_shape)
    layer_boundary_cost=layer_boundary_cost2(clew,gradient_y,image)
    intersection_cost=clews_intersection_cost_while_merging(clew,existing_clew_matrix,image)
    return layer_boundary_cost+intersection_cost+high_variance_cost

if __name__ == '__main__':
    #################################################################
    gradient_y = gradient_image_y(image)
    #####################################Z############################
    grid_shape = (10, 3)                        ####  update
    grid_variances = gen_variance_grid(image, grid_shape)
    print(grid_variances)
    variance_max=np.max(grid_variances)
    clew_size_for_each_square = np.minimum((grid_variances/100) .astype(int), 6)
    if clew_size_for_each_square[0][0]==0:
        clew_size_for_each_square[0][0]=1


    horizontal_square_size=image.shape[1]/grid_shape[1]    

    verticle_square_size=image.shape[0]/grid_shape[0]
    print("clue size", clew_size_for_each_square)    
    for m in range(grid_shape[0]):
        for n in range(grid_shape[1]):
            print("m n",m, n)
            class Clew():
                def __init__(self, bounds, vector=None):
                    self.bounds = bounds.astype
                    if vector is not None:
                        self.vector = vector
                    else:
                        self.vector = np.random.uniform(bounds[:, 0], bounds[:, 1])
                    
                    self.worms = [Camo_Worm(*self.vector[i:i+8]) for i in range(0, len(self.vector), 8)]
                
                def __getitem__(self, key):
                    return self.worms[key]
                
                def __iter__(self):
                    return iter(self.worms)
                
                def __len__(self):
                    return len(self.worms)
                @staticmethod
                def generate_bounds(num_worms, imshape):
                    bounds = np.array([
                        [n*horizontal_square_size, (n+1)*horizontal_square_size-1],      # x
                        [m*verticle_square_size, (m+1)*verticle_square_size-1],      # y
                        [10, 600],            # r
                        [-0.3, 0.3],           # theta
                        [1, 20],             # dr
                        [-np.pi, np.pi],           # dgamma
                        [2, 3],              # width
                        [0, 255]              # colour
                    ])

                    return np.tile(bounds, (num_worms, 1))
                def generate_inibounds(num_worms, imshape):              # bounds for initial population
                    inibounds = np.array([
                        [n*horizontal_square_size, (n+1)*horizontal_square_size-1],      # x
                        [m*verticle_square_size, (m+1)*verticle_square_size-1],  
                        [10, 100],            # r
                        [-0.3, 0.3],           # theta
                        [1, 20],             # dr
                        [-np.pi, np.pi],           # dgamma
                        [2, 3],              # width
                        [0, 255]              # colour
                    ])
                    return np.tile(inibounds, (num_worms, 1))
            for i in range(clew_size_for_each_square[m][n]):                                                     # add worm n times 
                bounds = Clew.generate_bounds(NUM_WORMS, image.shape)
                inibounds = Clew.generate_inibounds(NUM_WORMS, image.shape)
                initial_population = [Clew(inibounds) for _ in range (6)]

                cost_fn = lambda x: final_cost(x, image)

                # Create an instance of the DE algorithm
                de = DE(
                    objective_function=cost_fn,
                    bounds=bounds,
                    initial_population=initial_population,
                    max_iter=200,        
                    F=0.1,
                    CR=0.5
                )

                while de.generation < de.max_iter:
                    de.iterate()
                    print(f"Generation {de.generation}: Best cost = {cost_fn(de.get_best())}")

                # Get the best clew
                best_clew = de.get_best()

                # Visualize the best clew from the final population
                # drawing = util.Drawing(image)
                # drawing.add_worms(best_clew.worms)
                # drawing.show()

                if i == 0 and m==0 and n==0:
                    current_clew = best_clew
                else:
                    current_clew.worms.extend(best_clew.worms)
                    print("i=",i)
                    # drawing = util.Drawing(image)
                    # drawing.add_worms(current_clew.worms)
                    # drawing.show()
                for worm in best_clew.worms:
                    existing_clew_matrix = add_worms_to_existing_clew(existing_clew_matrix, worm_matrix(worm, image))
                    print(np.sum(existing_clew_matrix))
        drawing = util.Drawing(image)
        drawing.add_worms(current_clew.worms)
        drawing.show()
    drawing = util.Drawing(image)
    drawing.add_worms(current_clew.worms)
    drawing.show()