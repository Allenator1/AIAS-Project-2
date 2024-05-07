import numpy as np
import cv2
import os

from optimize import *
import util


if __name__ == "__main__":
	image_dir = "Mendeley dataset"
	images = os.listdir(image_dir)
	os.makedirs("output/recursive", exist_ok=True)
	os.makedirs("output/multiscale", exist_ok=True)
	os.makedirs("output/iterative", exist_ok=True)

	for image_name in images:
		image_path = os.path.join(image_dir, image_name)
		print(f"Processing {image_path}")
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		height, width = image.shape
		image = np.flipud(image)

		# Recursive optimisation
		final_worms = recursive_subdivision_optimisation(image, max_depth=4)
		drawing = util.Drawing(image)
		drawing.add_worms(final_worms)
		drawing.show(save=f"output/recursive/{image_name}")

		# Multiscale optimisation
		final_worms = multiscale_optimisation(image)
		drawing = util.Drawing(image)
		drawing.add_worms(final_worms)
		drawing.show(save=f"output/recursive/{image_name}")

		# Multiscale optimisation
		final_worms = iterative_optimisation(image, num_iter=20)
		drawing = util.Drawing(image)
		drawing.add_worms(final_worms)
		drawing.show(save=f"output/recursive/{image_name}")

