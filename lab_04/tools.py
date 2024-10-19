from typing import Any, NewType, Tuple
import numpy as np
from PIL import Image
import cv2


ImageType = NewType("ImageType", np.ndarray[Any, np.dtype[np.uint8]])


def to_grayscale(image: ImageType) -> ImageType:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_binary(gradient_image: ImageType, threshold: int) -> ImageType:
    _, binary_image = cv2.threshold(gradient_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def apply_filter(image: ImageType, kernel: ImageType) -> ImageType:
    height, width = image.shape
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode="constant", constant_values=0)

    result = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            window = padded_image[i : i + kernel_size, j : j + kernel_size]
            result[i, j] = np.sum(window * kernel)

    return result


def normalize(image: ImageType) -> ImageType:
    return np.clip(
        (image - image.min()) / (image.max() - image.min()) * 255, 0, 255
    ).astype(np.uint8)


def apply_scharr(image: ImageType) -> Tuple[ImageType, ImageType, ImageType]:
    Gx_kernel = ImageType(np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]]))
    Gy_kernel = ImageType(np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]))

    Gx = apply_filter(image, Gx_kernel)
    Gy = apply_filter(image, Gy_kernel)
    G = ImageType(np.sqrt(Gx**2 + Gy**2))

    Gx_normalized = normalize(Gx)
    Gy_normalized = normalize(Gy)
    G_normalized = normalize(G)

    return Gx_normalized, Gy_normalized, G_normalized


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
