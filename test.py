import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import skimage.restoration as sr

from optimize import *
import util


def matplotlib_diagrams(original_dir, camoflaged_dir, uncamoflaged_dir):
    """
    Create matplotlib diagrams for the original, camoflaged and uncamoflaged images.
    """
    original_images = os.listdir(original_dir)
    os.makedirs("Figures", exist_ok=True)

    for image_name in original_images:
        original_image = cv2.imread(os.path.join(original_dir, image_name), cv2.IMREAD_GRAYSCALE)
        camoflaged_image = cv2.imread(os.path.join(camoflaged_dir, image_name), cv2.IMREAD_GRAYSCALE)
        uncamoflaged_image = cv2.imread(os.path.join(uncamoflaged_dir, image_name), cv2.IMREAD_GRAYSCALE)

        sigma_original = sr.estimate_sigma(original_image, channel_axis=None)
        sigma_camoflaged = sr.estimate_sigma(camoflaged_image, channel_axis=None)

        _, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(original_image, cmap="gray")
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(camoflaged_image, cmap="gray")
        axs[1].set_title("Camouflaged")
        axs[1].axis("off")

        axs[2].imshow(uncamoflaged_image, cmap="gray")
        axs[2].set_title("Uncamouflaged")
        axs[2].axis("off")
        
        plt.suptitle(f"Sigma original: {sigma_original:.2f}, Sigma camouflaged: {sigma_camoflaged:.2f}")
        plt.savefig(f"Figures/{image_name}", bbox_inches="tight")


def tune_hyperparameters(image_dir, image_name):
    """
    Create a CSV of the PSNR and Sigma values for different sets of hyperparameters.
    """
    impath = os.path.join(image_dir, image_name)
    image = np.flipud(cv2.imread(impath, cv2.IMREAD_GRAYSCALE))


    # Generate the optimal worms
    F_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    CR_values = [0.2, 0.4, 0.6, 0.8, 1.0]

    os.makedirs("output/hyperparameter_tuning", exist_ok=True)

    for f in F_values:
        PARAMS["F"] = f
        for cr in CR_values:
            PARAMS["CR"] = cr
            print(f"Optimising for F={f}, CR={cr}")
            final_worms = multiscale_optimisation(image)
            final_worms.extend(recursive_subdivision_optimisation(image, max_depth=4))
            drawing = util.Drawing(image)
            drawing.add_worms(final_worms)
            drawing.show(save=f"output/hyperparameter_tuning/{image_name}_F{f}_CR{cr}.png")


def measure_hyperparameter_performance(param_dir, original_impath):
    """
    Measure the performance of the hyperparameters in the param_dir.
    """
    original = cv2.imread(original_impath, cv2.IMREAD_GRAYSCALE)

    F_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    CR_values = [0.2, 0.4, 0.6, 0.8, 1.0]

    csv = open("output/hyperparameter_performance.csv", "w")

    for f in F_values:
        for cr in CR_values:
            camoflaged_impath = f"{param_dir}/original.png_F{f}_CR{cr}.png"
            camoflaged = cv2.imread(camoflaged_impath, cv2.IMREAD_GRAYSCALE)
            sigma_camoflaged = sr.estimate_sigma(camoflaged, channel_axis=None)
            csv.write(f"{f},{cr},{sigma_camoflaged}\n")
            

def denoise_all_images(image_dir="Mendeley dataset", output_dir="output/camouflaged"):
    """
    Denoise all images in the image_dir and save them in the output_dir.
    """
    images = os.listdir(image_dir)
    os.makedirs(output_dir, exist_ok=True)

    for image_name in images:
        image_path = os.path.join(image_dir, image_name)
        print(f"Processing {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.flipud(image)

        # Multiscale optimisation
        print("Multiscale optimisation")
        final_worms = multiscale_optimisation(image)
        print("Recursive subdivision optimisation")
        final_worms.extend(recursive_subdivision_optimisation(image, max_depth=4))
        drawing = util.Drawing(image)
        drawing.add_worms(final_worms)
        drawing.show(save=f"{output_dir}/{image_name}")


if __name__ == "__main__":
    measure_hyperparameter_performance("output/hyperparameter_tuning", "Mendeley dataset/original.png")
    # tune_hyperparameters("Mendeley dataset", "original.png")

