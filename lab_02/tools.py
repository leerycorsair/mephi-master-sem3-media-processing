from typing import Any, NewType
import numpy as np
from PIL import Image
import cv2


ImageType = NewType("ImageType", np.ndarray[Any, np.dtype[np.uint8]])

RED_WEIGHT = 0.299
GREEN_WEIGHT = 0.587
BLUE_WEIGHT = 0.114


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


def niblack_binarization(image: ImageType, window_size: int, k: float) -> ImageType:
    binary_image = np.zeros_like(image, dtype=np.uint8)

    mean = cv2.boxFilter(
        image.astype(np.float32), ddepth=-1, ksize=(window_size, window_size)
    )
    mean_sq = cv2.boxFilter(
        image.astype(np.float32) ** 2, ddepth=-1, ksize=(window_size, window_size)
    )
    variance = mean_sq - mean**2
    stddev = np.sqrt(np.maximum(variance, 0))

    threshold = mean + k * stddev
    binary_image[image >= threshold] = 255
    binary_image[image < threshold] = 0

    return ImageType(binary_image)


def rgb_to_grayscale(image: ImageType) -> ImageType:
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    grayscale = (
        RED_WEIGHT * red_channel
        + GREEN_WEIGHT * green_channel
        + BLUE_WEIGHT * blue_channel
    ).astype(np.uint8)

    return ImageType(grayscale)
