import numpy as np
import cv2
import os

from optimize import *
import util


if __name__ == "__main__":
	image_dir = "Mendeley dataset"
	# images = os.listdir(image_dir)
	images = ['original.png', 'DRUSEN-1083159-1.jpeg', 'DRUSEN-1219727-1.jpeg', 'CNV-1016042-3.jpeg', 'DRUSEN-1173253-1.jpeg', 'CNV-1016042-2.jpeg', 
		   'DME-1102486-2.jpeg', 'CNV-1016042-1.jpeg', 'CNV-103044-10.jpeg', 'DME-1102486-1.jpeg', 'DRUSEN-1193659-1.jpeg']
	os.makedirs("output/multiscale", exist_ok=True)
	os.makedirs("output/iterative", exist_ok=True)

	for image_name in images:
		image_path = os.path.join(image_dir, image_name)
		print(f"Processing {image_path}")
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		height, width = image.shape
		image = np.flipud(image)

		# Multiscale optimisation
		print("Multiscale optimisation")
		final_worms = multiscale_optimisation(image)
		print("Recursive subdivision optimisation")
		final_worms.extend(recursive_subdivision_optimisation(image, max_depth=5))
		drawing = util.Drawing(image)
		drawing.add_worms(final_worms)
		drawing.show(save=f"output/multiscale/{image_name}")

		# Iterative optimisation
		print("Iterative optimisation")
		final_worms = iterative_optimisation(image, num_iter=20)
		drawing = util.Drawing(image)
		drawing.add_worms(final_worms)
		drawing.show(save=f"output/iterative/{image_name}")

