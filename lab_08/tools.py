import os
from typing import Any, NewType

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import feature, io, color

DISTANCE = 2
ANGLES = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

ImageType = NewType("ImageType", np.ndarray[Any, np.dtype[np.uint8]])


def grayscale(image: ImageType) -> ImageType:
    if image.ndim == 3:
        gray_image = color.rgb2gray(image)
        gray_image = (gray_image * 255).astype(np.uint8)
    else:
        gray_image = image

    return gray_image


def load_image(input_path: str) -> np.ndarray:
    try:
        image = io.imread(input_path)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        raise


def contrast(glcm):
    return feature.graycoprops(glcm, prop="contrast")


def local(glcm):
    return feature.graycoprops(glcm, prop="homogeneity")


def gamma_correction(image: ImageType, gamma: float) -> ImageType:
    image_normalized = image / 255.0
    corrected_image = np.power(image_normalized, gamma)
    corrected_image = (corrected_image * 255).astype(np.uint8)

    return corrected_image


def save_metrics_to_file(metrics: dict, output_dir: str, filename: str) -> None:
    metrics_file = os.path.join(output_dir, f"{filename}_metrics.txt")
    with open(metrics_file, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def haralick_matrix(image: ImageType, distance=DISTANCE, angles=ANGLES):
    glcm = feature.graycomatrix(
        image, distances=[distance], angles=angles, symmetric=True, normed=True
    )
    glcm_log = np.log1p(glcm)

    return glcm_log


def save_glcm_images(glcm: np.ndarray, output_dir: str, filename: str) -> None:
    for idx, angle in enumerate([0, 90, 180, 270]):
        glcm_angle = glcm[:, :, 0, idx]

        glcm_file = os.path.join(output_dir, f"{filename}_glcm_{angle}deg.txt")
        np.savetxt(glcm_file, glcm_angle, fmt="%f")

        plt.imshow(glcm_angle, cmap="grey")
        plt.title(f"GLCM для угла {angle}°")
        plt.colorbar()
        glcm_img_path = os.path.join(output_dir, f"{filename}_glcm_{angle}deg.png")
        plt.savefig(glcm_img_path)
        plt.close()


def save_image(image: ImageType, output_path: str) -> None:
    img = Image.fromarray(image)
    img.save(output_path)
