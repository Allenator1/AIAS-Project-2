import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.bezier as mbezier
from PIL import Image

Path = mpath.Path
rng = np.random.default_rng()

class Camo_Worm:
    """
    Class representing a camouflage worm.
    """
    def __init__(self, x, y, r, theta, deviation_r, deviation_gamma, width, colour):
        self.x = x
        self.y = y
        self.r = r
        self.theta = theta
        self.dr = deviation_r
        self.dgamma = deviation_gamma
        self.width = width
        self.colour = colour

        p0 = [self.x - self.r * np.cos(self.theta), self.y - self.r * np.sin(self.theta)]
        p2 = [self.x + self.r * np.cos(self.theta), self.y + self.r * np.sin(self.theta)]
        p1 = [self.x + self.dr * np.cos(self.theta + self.dgamma), self.y + self.dr * np.sin(self.theta + self.dgamma)]
        self.bezier = mbezier.BezierSegment(np.array([p0, p1, p2]))

    def control_points(self):
        """Get control points of the Bezier curve."""
        return self.bezier.control_points

    def path(self):
        """Get the path of the worm."""
        return mpath.Path(self.control_points(), [Path.MOVETO, Path.CURVE3, Path.CURVE3])

    def patch(self):
        """Get the patch of the worm."""
        if isinstance(self.colour, tuple):  # Check if colour is a tuple
            rgba_color = (self.colour[0] / 255, self.colour[1] / 255, self.colour[2] / 255, 1.0)  # Convert colour tuple to RGBA format
        else:
            rgba_color = (self.colour / 255, self.colour / 255, self.colour / 255, 1.0)  # Convert scalar to RGBA format
        return mpatches.PathPatch(self.path(), fc='None', ec=rgba_color, lw=self.width, capstyle='round')

    def intermediate_points(self, intervals=None):
        """
        Get intermediate points along the worm's path.

        Args:
            intervals (int): Number of intervals.

        Returns:
            numpy.ndarray: Array of intermediate points.
        """
        if intervals is None:
            intervals = max(3, int(np.ceil(self.r / 8)))
        return self.bezier.point_at_t(np.linspace(0, 1, intervals))

    def approx_length(self):
        """Approximate the length of the worm."""
        intermediates = self.intermediate_points()
        intermediates = np.array(intermediates)
        eds = np.linalg.norm(intermediates[1:] - intermediates[:-1], axis=1)
        return np.sum(eds)

    def colour_at_t(self, t, image):
        """Get color of the worm at t."""
        intermediates = np.int64(np.round(np.array(self.bezier.point_at_t(t)).reshape(-1, 2)))
        colours = [image[point[0], point[1]] for point in intermediates]
        return (np.array(colours) / 255)

    def get_params(self):
        """Get worm parameters."""
        return [self.x, self.y, self.r, self.theta, self.dr, self.dgamma, self.width, self.colour]

    @staticmethod
    def random_worm(imshape, init_params):
        """Generate a random worm."""
        (radius_std, deviation_std, width_theta) = init_params
        (ylim, xlim) = imshape
        midx = xlim * rng.random()
        midy = ylim * rng.random()
        r = radius_std * np.abs(rng.standard_normal())
        theta = rng.random() * np.pi
        dr = deviation_std * np.abs(rng.standard_normal())
        dgamma = rng.random() * np.pi
        colour = tuple(rng.integers(0, 256, size=3))  # Random RGB tuple
        width = width_theta * rng.standard_gamma(3)
        return Camo_Worm(midx, midy, r, theta, dr, dgamma, width, colour)

    @staticmethod
    def random_clew(size, imshape, init_params):
        """Generate a random clew of worms."""
        clew = []
        for i in range(size):
            clew.append(Camo_Worm.random_worm(imshape, init_params))
        return clew

class DE:
    """
    Differential Evolution algorithm.
    """
    def __init__(self, objective_function, bounds, NP=10, max_iter=1000, F=0.5, CR=0.9):
        self.objective_function = objective_function
        self.bounds = bounds
        self.NP = NP
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        self.population = [self.random_worm(bounds) for _ in range(NP)]
        self.generation = 0
        self.all_worms = []  # To store all worms generated

    def random_worm(self, bounds):
        """Generate a random worm within bounds."""
        worm_params = np.random.uniform(bounds[:, 0], bounds[:, 1])
        return Camo_Worm(*worm_params)

    def iterate(self):
        """Perform a single iteration of DE."""
        for i in range(self.NP):
            # Select three random indices
            r1, r2, r3 = np.random.choice(self.NP, 3, replace=False)
            target = self.population[i]

            # Mutation
            mutant_params = np.array(target.get_params()) + self.F * (
                np.array(self.population[r1].get_params()) - np.array(self.population[r2].get_params())
            )

            # Crossover
            cross_points = np.random.rand(len(mutant_params)) < self.CR
            trial_params = np.where(cross_points, mutant_params, target.get_params())

            # Clip to bounds
            trial_params = np.clip(trial_params, self.bounds[:, 0], self.bounds[:, 1])

            # Create a new Camo_Worm instance
            trial = Camo_Worm(*trial_params)

            # Selection
            if self.objective_function(trial) < self.objective_function(target):
                self.population[i] = trial

            self.all_worms.append(trial)  # Add trial worm to all_worms

        self.generation += 1

    def get_best(self):
        """Get the best worm in the population."""
        return min(self.population, key=self.objective_function)

def simplified_cost(worm, image):
    """
    Simplified cost function.
    """
    # Get the intermediate points along the worm's path
    intermediate_points = worm.intermediate_points()

    # Calculate the difference between the worm's length and the desired length (e.g., 100 pixels)
    length_diff = np.abs(worm.approx_length() - 100)

    # Calculate the deviation from a straight line
    deviation = np.linalg.norm(intermediate_points[-1] - intermediate_points[0])

    # Combine the length difference and deviation into a single cost value
    cost = length_diff + 0.1 * deviation  # Adjust the weight as needed

    return cost

# Load the image
image_dir = 'images'
image_name = 'zibra.jpg'
image = np.array(Image.open(f"{image_dir}/{image_name}"))

# Define the bounds for the worm parameters
bounds = np.array([
    [0, image.shape[1]],  # x
    [0, image.shape[0]],  # y
    [10, 100],            # r
    [0, 2 * np.pi],       # theta
    [10, 50],             # dr
    [0, 2 * np.pi],       # dgamma
    [1, 10],              # width
    [0, 255]              # colour
])

# Create an instance of the DE algorithm
de = DE(
    objective_function=lambda worm: simplified_cost(worm, image),
    bounds=bounds,
    NP=50,                # Adjust population size
    max_iter=100,         # Increase number of iterations
    F=0.5,
    CR=0.9
)

# Run the DE algorithm
while de.generation < de.max_iter:
    de.iterate()
    print(f"Generation {de.generation}: Best cost = {simplified_cost(de.get_best(), image)}")

# Get the best camouflage worm
best_worm = de.get_best()

# # Visualize all worms on every iteration
# fig, ax = plt.subplots()
# ax.imshow(image)
# for worm in de.all_worms:  # Plot all worms
#     ax.add_patch(worm.patch())
# plt.axis('off')
# plt.show()

# Visualize the best worm from the final population
fig, ax = plt.subplots()
ax.imshow(image)
for worm in de.population:  # Plot only worms from the final population
    ax.add_patch(worm.patch())
plt.axis('off')
plt.show()
