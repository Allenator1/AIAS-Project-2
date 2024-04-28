'''
High level approach on worm density component

To design an objective function that penalizes areas with too many overlapping worms, we will need to quantify the degree of overlap between worms in the generated image. One way to approach this is to calculate the density of worms in different regions of the image and penalize regions with high worm density.

1. Define Worm Density Metric: Define a metric to quantify the density of worms in different regions of the image. This could be based on the number of worms overlapping in a particular region or the proximity of worms to each other.
2. Preprocessing and Image Generation: Preprocess the original OCT retina image and generate a new image by overlaying camo worms, similar to the previous step.
3. Objective Function: Design an objective function that evaluates the density of worms in the generated image and penalizes regions with high worm density. This function should take the new image with camo worms as input and return a scalar value representing the degree of penalization.
4. Implementation: Implement the objective function in Python. This function should process the image with camo worms and calculate the worm density metric.
5. Integration with DE Algorithm: Integrate the objective function with the DE algorithm by passing it as a parameter to the DE class constructor.
'''
'''
# simple objective function to calculate the density of worms based on the distance between worm
def simplified_cost_worm_density(clew, image):
    """Calculate the cost of a clew of worms."""
    worm_density = 0
    for i in range(len(clew)):
        for j in range(i+1, len(clew)):
            worm_density += np.linalg.norm(clew[i].get_params()[:2] - clew[j].get_params()[:2]) #distance between worm centers by calculating the Euclidean distance between the centre-points of the worms
    worm_density /= image.size
    return worm_density
'''

# simple objective function to penalise high desinty of worms in different regions of the image with grid
def simplified_cost_worm_density(clew, image):
    """Calculate the cost of a clew of worms."""
    worm_density = 0
    grid_size = 10
    grid = np.zeros((grid_size, grid_size)) #create a grid to divide the image into different regions
    for worm in clew:
        x, y = worm.get_params()[:2]
        grid_x = int(x / image.width * grid_size) #calculate the grid position of the worm
        grid_y = int(y / image.height * grid_size)
        grid[grid_x, grid_y] += 1 #increment the count of worms in the corresponding grid cell
    for i in range(grid_size):
        for j in range(grid_size):
            worm_density += grid[i, j] * (grid[i, j] - 1) #calculate the density of worms in each grid cell
    worm_density /= image.size
    return worm_density








#option 2
'''# simple objective function for the optimise algorithm that penalises high worm density based on the approximity of worm centres
def simplified_cost_worm_density(clew, image):
    """Calculate the cost of a clew of worms."""
    worm_density = 0
    for worm in clew:
        worm_density += worm.width * worm.r
    worm_density /= image.size
    return worm_density
'''







