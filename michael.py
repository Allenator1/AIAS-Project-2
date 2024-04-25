# simple objective function for the optimise algorithm that penalises high worm density based on the approximity of worms
def simplified_cost_worm_density(clew, image):
    """Calculate the cost of a clew of worms."""
    worm_density = 0
    for worm in clew:
        worm_density += worm.width * worm.r
    worm_density /= image.size
    return worm_density






