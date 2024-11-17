from typing import Any, NewType
import numpy as np
from PIL import Image
import cv2

ImageType = NewType("ImageType", np.ndarray[Any, np.dtype[np.uint8]])


def to_grayscale(image: ImageType) -> ImageType:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_scharr(image: ImageType) -> ImageType:
    if not isinstance(image, np.ndarray):
        raise TypeError("Входное изображение должно быть типа numpy.ndarray.")

    if len(image.shape) != 2:
        raise ValueError("Изображение должно быть в градациях серого (2D массив).")

    scharr_x_kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64)

    scharr_y_kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float64)

    scharr_x = convolve2d(image, scharr_x_kernel)
    scharr_y = convolve2d(image, scharr_y_kernel)

    magnitude = np.sqrt(scharr_x**2 + scharr_y**2)

    magnitude = (magnitude / magnitude.max()) * 255
    magnitude = magnitude.astype(np.uint8)

    return ImageType(magnitude)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(
        image,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode="constant",
        constant_values=0,
    )
    convolved = np.zeros_like(image, dtype=np.float64)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i : i + kernel_height, j : j + kernel_width]
            convolved[i, j] = np.sum(region * kernel)

    return convolved


def load_image(input_path: str) -> ImageType:
    try:
        image = Image.open(input_path).convert("RGB")
        return ImageType(np.array(image))
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        raise


def save_image(image: ImageType, output_path: str) -> None:
    img = Image.fromarray(image)
    img.save(output_path)
