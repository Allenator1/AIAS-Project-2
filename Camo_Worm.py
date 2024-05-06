import numpy as np

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.bezier as mbezier
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

import util

from differential_evolution.DE import Target


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
        
        worm_length = self.approx_length()
        n_intervals = int(worm_length // self.width)
        if n_intervals > 0:
            start = 1 / (2 * n_intervals); end = 1 - 1 / (2 * n_intervals)
            self.window_points = self.bezier.point_at_t(np.linspace(start, end, n_intervals))
        else:
            self.window_points = [self.bezier.point_at_t(0.5)]

    def control_points(self):
        """Get control points of the Bezier curve."""
        return self.bezier.control_points

    def path(self):
        """Get the path of the worm."""
        return mpath.Path(self.control_points(), [Path.MOVETO, Path.CURVE3, Path.CURVE3])

    def patch(self):
        """Get the patch of the worm."""
        rgb_col = (self.colour / 255, self.colour / 255, self.colour / 255)
        return mpatches.PathPatch(self.path(), fc='None', ec="White", lw=self.width, capstyle='round')

    def intermediate_points(self, intervals=None):
        """
        Get intermediate points along the worm's path.

        Args:
            intervals (int): Number of intervals.

        Returns:
            numpy.ndarray: Array of intermediate points.
        """
        if intervals is None:
            intervals = max(100, int(np.ceil(self.r / 8)))
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
        return np.mean(colours) / 255  # Calculate grayscale value

    def get_params(self):
        """Get worm parameters."""
        return np.array([self.x, self.y, self.r, self.theta, self.dr, self.dgamma, self.width, self.colour])
    
    def get_worm_vals(self, image):
        worm_vals = []
        for x, y in self.window_points:
            xstart = max(0, x - self.width // 2)
            xend = min(x + self.width // 2, image.shape[1])
            ystart = max(0, y - self.width // 2)
            yend = min(y + self.width // 2, image.shape[0])
            vals = np.reshape(image[int(ystart):int(yend), int(xstart):int(xend)], -1)
            worm_vals.append(vals)
        return np.concatenate(worm_vals)

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
        colour = rng.integers(0, 256)  # Random grayscale value
        width = width_theta * rng.standard_gamma(3)
        return Camo_Worm(midx, midy, r, theta, dr, dgamma, width, colour)
    
    @staticmethod
    def random_clew(size, imshape, init_params):
        """Generate a random clew of worms."""
        clew = []
        for i in range(size):
            clew.append(Camo_Worm.random_worm(imshape, init_params))
        return clew
    

class Clew():
    def __init__(self, bounds, vector=None):
        self.bounds = bounds
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
            [200, 400],      # x
            [300, 530],      # y
            [10, 600],            # r
            [-0.3, 0.3],           # theta
            [1, 20],             # dr
            [-np.pi, np.pi],           # dgamma
            [2, 4],              # width
            [0, 255]              # colour
        ])

        return np.tile(bounds, (num_worms, 1))
    def generate_inibounds(num_worms, imshape):              # bounds for initial population
        inibounds = np.array([
           [200.2, 400],      # x
            [400, 530],      # y
            [10, 100],            # r
            [-0.3, 0.3],           # theta
            [1, 20],             # dr
            [-np.pi, np.pi],           # dgamma
            [2, 4],              # width
            [0, 255]              # colour
        ])
        return np.tile(inibounds, (num_worms, 1))


## TEST USAGE
if __name__ == "__main__":
    image_dir = 'images'
    image_name='original'
    mask = [320, 560, 160, 880] 	# ymin ymax xmin xmax
    
    image = util.prep_image(image_dir, image_name, mask)
    
    worm = Camo_Worm.random_worm(image.shape[:2], (40, 30, 5))
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray', origin='lower')
    ax.add_patch(worm.patch())

    worm_vals = worm.get_worm_vals(image)
    print(worm_vals)
    

