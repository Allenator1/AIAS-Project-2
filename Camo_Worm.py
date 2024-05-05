import numpy as np
import time
import cv2

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
    def __init__(self, bounds, vector=None):
        if vector is not None:
            self.vector = vector
        else:
            self.vector = np.random.uniform(bounds[:, 0], bounds[:, 1])
        x, y, r, theta, dr, dgamma = self.vector
        self.colour = 255
        self.width = 2

        (xmin, xmax), (ymin, ymax) = bounds[0:2]

        p0 = [x - r * np.cos(theta), y - r * np.sin(theta)]
        p2 = [x + r * np.cos(theta), y + r * np.sin(theta)]
        p1 = [x + dr * np.cos(theta + dgamma), y + dr * np.sin(theta + dgamma)]

        control_points = np.array([p0, p1, p2])
        control_points = np.clip(control_points, [xmin, ymin], [xmax, ymax])

        self.bezier = mbezier.BezierSegment(control_points)
        num_points = int(self.approx_length())
        points = self.intermediate_points(num_points)
        points = np.int64(np.round(points))
        [x_points, y_points] = np.unique(points, axis=0).T
        self.indices = tuple([y_points, x_points])
        

    def control_points(self):
        """Get control points of the Bezier curve."""
        return self.bezier.control_points

    def path(self):
        """Get the path of the worm."""
        return mpath.Path(self.control_points(), [Path.MOVETO, Path.CURVE3, Path.CURVE3])

    def patch(self):
        """Get the patch of the worm."""
        rgb_col = (self.colour / 255, self.colour / 255, self.colour / 255)
        return mpatches.PathPatch(self.path(), fc='None', ec=rgb_col, lw=self.width, capstyle='round')

    def intermediate_points(self, intervals=3):
        """
        Get intermediate points along the worm's path.

        Args:
            intervals (int): Number of intervals.

        Returns:
            numpy.ndarray: Array of intermediate points.
        """
        return self.bezier.point_at_t(np.linspace(0, 1, intervals))

    def approx_length(self, intervals=3):
        """Approximate the length of the worm."""
        intermediates = self.intermediate_points(intervals)
        intermediates = np.array(intermediates)
        eds = np.linalg.norm(intermediates[1:] - intermediates[:-1], axis=1)
        return np.sum(eds)

    def colour_at_t(self, t, image):
        """Get color of the worm at t."""
        intermediates = np.int64(np.round(np.array(self.bezier.point_at_t(t)).reshape(-1, 2)))
        colours = [image[point[0], point[1]] for point in intermediates]
        return np.mean(colours) / 255  # Calculate grayscale value

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
        bounds = Camo_Worm.generate_bounds([0, xlim, 0, ylim])
        worm = Camo_Worm(bounds, np.array([midx, midy, r, theta, dr, dgamma]))
        return worm
    
    @staticmethod
    def random_clew(size, imshape, init_params):
        """Generate a random clew of worms."""
        clew = []
        for i in range(size):
            clew.append(Camo_Worm.random_worm(imshape, init_params))
        return clew
    
    @staticmethod
    def generate_bounds(worm_bounds):
        x1, x2, y1, y2 = worm_bounds
        dim_length = max(x2 - x1, y2 - y1)
        r_min = dim_length / 10
        r_max = dim_length 

        return np.array([
            [x1, x2 - 1],         # x
            [y1, y2 - 1],         # y
            [r_min, r_max],       # r
            [0, np.pi],           # theta
            [10, r_min],          # dr
            [0, np.pi],           # dgamma
        ])


## TEST USAGE
if __name__ == "__main__":
    image_dir = 'images'
    image_name='original'
    mask = [320, 560, 160, 880] 	# ymin ymax xmin xmax
    
    image = cv2.imread(f"{image_dir}/{image_name}.png", cv2.IMREAD_GRAYSCALE)
    image = image[mask[0]:mask[1], mask[2]:mask[3]]

    t1 = time.time()
    worm = Camo_Worm.random_worm(image.shape[:2], (100, 30, 10))
    t2 = time.time()
    print(f"Time taken: {t2 - t1} s")
    


