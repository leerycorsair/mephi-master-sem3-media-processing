from typing import Any, NewType
import numpy as np
from PIL import Image
import cv2


ImageType = NewType("ImageType", np.ndarray[Any, np.dtype[np.uint8]])


def binarize(image: ImageType) -> ImageType:
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 1, cv2.THRESH_BINARY)
    return binary_image


def load_image(input_path: str) -> ImageType:
    try:
        image = Image.open(input_path).convert("RGB")
        return ImageType(np.array(image))
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        raise


def filter_image(binary_image: ImageType, k: int, aperture_size: int = 3) -> ImageType:
    result_image = np.zeros_like(binary_image)

    height, width = binary_image.shape

    for i in range(height):
        for j in range(width):
            i_min = max(0, i - aperture_size // 2)
            i_max = min(height, i + aperture_size // 2 + 1)
            j_min = max(0, j - aperture_size // 2)
            j_max = min(width, j + aperture_size // 2 + 1)

            aperture = binary_image[i_min:i_max, j_min:j_max]
            num_ones = np.sum(aperture)
            num_zeros = aperture.size - num_ones

            if num_ones >= k:
                result_image[i, j] = 1
            elif num_zeros >= aperture.size + 1 - k:
                result_image[i, j] = 0
            else:
                result_image[i, j] = binary_image[i, j]

    color_result_image = cv2.cvtColor(result_image * 255, cv2.COLOR_GRAY2RGB)

    return ImageType(color_result_image)


def xor(image1: ImageType, image2: ImageType) -> ImageType:
    if image1.shape != image2.shape:
        raise ValueError(
            "Изображения должны иметь одинаковые размеры для выполнения XOR."
        )

    xor_image = cv2.bitwise_xor(image1, image2)
    return ImageType(xor_image)


def save_image(image: ImageType, output_path: str) -> None:
    img = Image.fromarray(image)
    img.save(output_path)
