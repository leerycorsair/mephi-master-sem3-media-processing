import numpy as np
from PIL import Image
from typing import Any, Optional, NewType
from tqdm import tqdm

ImageType = NewType("ImageType", np.ndarray[Any, np.dtype[np.uint8]])


def load_image(input_path: str) -> Optional[ImageType]:
    try:
        image = Image.open(input_path).convert("RGB")
        return ImageType(np.array(image))
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return None


def save_image(image: ImageType, output_path: str) -> None:
    img = Image.fromarray(image)
    img.save(output_path)


def stretch_image(image: ImageType, m: float) -> ImageType:
    old_height, old_width, _ = image.shape
    new_height = int(old_height * m)
    new_width = int(old_width * m)

    stretched_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in tqdm(range(new_height)):
        for x in range(new_width):
            old_y = int(y / m)
            old_x = int(x / m)
            stretched_image[y, x] = image[old_y, old_x]

    return ImageType(stretched_image)


def compress_image(image: ImageType, n: float) -> ImageType:
    old_height, old_width, _ = image.shape
    new_height = int(old_height / n)
    new_width = int(old_width / n)

    compressed_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in tqdm(range(new_height)):
        for x in range(new_width):
            old_y = int(y * n)
            old_x = int(x * n)
            compressed_image[y, x] = image[old_y, old_x]

    return ImageType(compressed_image)


def resample_image_two_pass(image: ImageType, m: float, n: float) -> ImageType:
    stretched_image = stretch_image(image, m)
    resampled_image = compress_image(stretched_image, n)
    return resampled_image


def resample_image_one_pass(image: ImageType, k: float) -> ImageType:
    old_height, old_width, _ = image.shape
    new_height = int(old_height * k)
    new_width = int(old_width * k)

    resampled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in tqdm(range(new_height)):
        for x in range(new_width):
            old_y = int(y / k)
            old_x = int(x / k)
            resampled_image[y, x] = image[old_y, old_x]

    return ImageType(resampled_image)
