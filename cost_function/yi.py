import numpy as np
from matplotlib.transforms import Bbox
import numpy as np

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

def worm_matrix(worm, image):   #matrix with 1 and 0. 1 indicate worm. 0 indicate empty space.
    worm_matrix = np.zeros_like(image, dtype=int)
    intermediate_points = worm.intermediate_points()
    intermediate_points = np.round(intermediate_points).astype(int)
    intermediate_points[:, 0] = np.clip(intermediate_points[:, 0], 0, image.shape[1] - 1) 
    intermediate_points[:, 1] = np.clip(intermediate_points[:, 1], 0, image.shape[0] - 1) 
    worm_matrix[intermediate_points[:, 1], intermediate_points[:, 0]] = 1
    return worm_matrix

def binary_image_into_grid(image, grid_shape): #compress worm_matrix to grid_shape
    h, w = image.shape
    grid_h, grid_w = grid_shape
    grid_binary_matrix = np.zeros(grid_shape)
    for i in range(grid_h):
        for j in range(grid_w):
            ymin = i * (h // grid_h)
            ymax = (i + 1) * (h // grid_h)
            xmin = j * (w // grid_w)
            xmax = (j + 1) * (w // grid_w)
            grid = image[ymin:ymax, xmin:xmax]
            grid_binary_matrix[i, j] = np.max(grid)
    return grid_binary_matrix

def adapt_to_variance(clew, grid_variances, image,grid_shape):
    worm_areas = []
    worm_mean_variances_list=[]
    for worm in clew:
        worm_matri=worm_matrix(worm, image)
        a=binary_image_into_grid(worm_matri, grid_shape)
        b=a*grid_variances
        if len(b[b != 0]) > 0:
            worm_neighbourhood_variance = np.mean(b[b != 0])
        else:
            worm_neighbourhood_variance = 0  
        worm_mean_variances_list.append(worm_neighbourhood_variance)
        length = worm.approx_length()
        width = worm.width
        area = length * width
        worm_areas.append(area)
    std_product = np.var([worm_number * worm_area for worm_number, worm_area in zip(worm_mean_variances_list
, worm_areas)])
    total_worm_area = np.sum(worm_areas)
    normalized_std = std_product / (total_worm_area**2)
    return normalized_std