import numpy as np
from matplotlib.transforms import Bbox


def gen_variance_grid(image, grid_shape):
    h, w = image.shape
    grid_h, grid_w = grid_shape
    grid_variances = np.zeros(grid_shape)
    for i in range(grid_h):
        for j in range(grid_w):
            ymin = i * (h // grid_h)
            ymax = (i + 1) * (h // grid_h)
            xmin = j * (w // grid_w)
            xmax = (j + 1) * (w // grid_w)
            grid = image[ymin:ymax, xmin:xmax]
            grid_variances[i, j] = int(np.var(grid))
    return grid_variances


def grids_intersected_by_worm(worm, grid_shape, image):
    grids_intersected = []
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            ymin = i * (image.shape[0] // grid_shape[0])
            ymax = (i + 1) * (image.shape[0] // grid_shape[0])
            xmin = j * (image.shape[1] // grid_shape[1])
            xmax = (j + 1) * (image.shape[1] // grid_shape[1])
            bbox = Bbox.from_extents(xmin, ymin, xmax, ymax)
            if worm.path().intersects_bbox(bbox):
                grids_intersected.append((i, j))
    return grids_intersected


def assign_worm_numbers(clew, grid_ratings, image):
    worm_numbers = []
    for worm in clew:
        intersected_grids = grids_intersected_by_worm(worm, grid_ratings.shape, image)
        if not intersected_grids:
            worm_numbers.append(0)
            continue
        worm_number = sum(grid_ratings[i][j] for i, j in intersected_grids) / len(intersected_grids)
        worm_numbers.append(worm_number)
    return worm_numbers


def adapt_to_variance(clew, grid_variance, image):
    """ 
    A cost function that drives worms to be shorter in areas of high variance,
    and longer in areas of low variance. 
    """
    worm_numbers = assign_worm_numbers(clew, grid_variance, image)
    worm_areas = []
    for worm in clew:
        length = worm.approx_length()
        width = worm.width
        area = length * width
        worm_areas.append(area)
    
    # Calculate the standard deviation of the product of worm numbers and areas
    std_product = np.std(np.array(worm_numbers) * np.array(worm_areas))
    
    # Calculate the sum of all worm areas
    total_worm_area = np.sum(worm_areas)
    
    # Normalize the standard deviation by dividing it by the total worm area, to encourage big clew size.
    normalized_std = std_product / total_worm_area
    
    return normalized_std


