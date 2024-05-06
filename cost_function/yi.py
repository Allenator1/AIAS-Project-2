import numpy as np
from matplotlib.transforms import Bbox
import numpy as np
from scipy.ndimage import median_filter

def gradient_image_y(input_image):
    # Convert the input image to floating point
    input_image = input_image.astype(np.float32)
    
    # Remove noise using median filter
    denoised_image = median_filter(input_image, size=3)
    
    # Compute gradients using central differences along the y-axis
    grad_y = np.gradient(denoised_image, axis=0)
    
    # Compute gradient magnitude (only along the y-axis)
    gradient_magnitude = np.abs(grad_y)
    return gradient_magnitude







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

def add_worms_to_existing_clew(existing_clew, matrix2):
    """
    Combine two binary matrices.

    Args:
        existing_clew (numpy.ndarray): Existing clew matrix.
        matrix2 (numpy.ndarray): Second binary matrix to add to the existing clew.

    Returns:
        numpy.ndarray: Updated clew matrix with added worms.
    """

    # Combine the matrices element-wise
    existing_clew = np.logical_or(existing_clew, matrix2).astype(np.uint8)

    return existing_clew



def worm_matrix(worm, image):  
    worm_matrix = np.zeros_like(image, dtype=int)
    intermediate_points = worm.intermediate_points()
    intermediate_points = np.round(intermediate_points).astype(int)
    intermediate_points[:, 0] = np.clip(intermediate_points[:, 0], 0, image.shape[1] - 1) 
    intermediate_points[:, 1] = np.clip(intermediate_points[:, 1], 0, image.shape[0] - 1) 
    worm_matrix[intermediate_points[:, 1], intermediate_points[:, 0]] = 1
    worm_matrix[intermediate_points[:, 1]+1, intermediate_points[:, 0]] = 1
    return worm_matrix


#compress worm_matrix to grid_shape
def binary_image_into_grid(image, grid_shape): 
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
            grid_binary_matrix[i, j] = 1 if np.any(grid) else 0
    return grid_binary_matrix


def layer_boundary_cost2(clew, gradient_y,image):
    total_costt = 0
    for worm in clew:
        worm_coverage_mask = worm_matrix(worm, image)
        worm_gradient_product = np.multiply(worm_coverage_mask, gradient_y)
        total_costt += np.sum(worm_gradient_product)
    return -1*int(total_costt)




def clews_intersection_cost_while_merging(new_clew, existing_clew_matrix,image):
    for worm in new_clew:
        merged_clew = np.sum( np.logical_and(worm_matrix(worm,image), existing_clew_matrix).astype(np.uint8))
    return 100000*merged_clew



# cost function to draw worms towards high variance area. 
def more_worms_to_high_variance_area(clew, grid_variances,image,grid_shape):
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
    return -1* np.mean(worm_mean_variances_list)