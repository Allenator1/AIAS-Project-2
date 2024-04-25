'''# simple objective function for the optimise algorithm that penalises high worm density based on the approximity of worm centres
def simplified_cost_worm_density(clew, image):
    """Calculate the cost of a clew of worms."""
    worm_density = 0
    for worm in clew:
        worm_density += worm.width * worm.r
    worm_density /= image.size
    return worm_density

'''

/ simple cost function that penalises high worm density based on distance of worms
def simplified_cost_worm_density(clew, image):
    """Calculate the cost of a clew of worms."""
    worm_density = 0
    for i in range(len(clew)):
        for j in range(i+1, len(clew)):
            worm_density += np.linalg.norm(clew[i].get_params()[:2] - clew[j].get_params()[:2])
    worm_density /= image.size
    return worm_density







