from PIL import Image, ImageDraw, ImageFont
import os
from typing import Callable, Optional, List

def create_image_with_text(
    letter: str,
    font: ImageFont.FreeTypeFont,
    style_func: Optional[
        Callable[[ImageDraw.ImageDraw, str, ImageFont.FreeTypeFont, int, int], None]
    ] = None,
    transform_func: Optional[Callable[[Image.Image], Image.Image]] = None,
) -> Image.Image:
    image_width, image_height = 200, 200
    image = Image.new("RGBA", (image_width, image_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), letter, font=font)

    text_center_x = (bbox[0] + bbox[2]) / 2
    text_center_y = (bbox[1] + bbox[3]) / 2

    image_center_x = image_width // 2
    image_center_y = image_height // 2

    x = image_center_x - text_center_x
    y = image_center_y - text_center_y

    if style_func:
        style_func(draw, letter, font, x, y)
    else:
        draw.text((x, y), letter, font=font, fill="black")

    if transform_func:
        image = transform_func(image)

    return image

def apply_bold(
    draw: ImageDraw.ImageDraw, letter: str, font: ImageFont.FreeTypeFont, x: int, y: int
) -> None:
    for offset in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        draw.text((x + offset[0], y + offset[1]), letter, font=font, fill="black")


def apply_italic_transform(image: Image.Image) -> Image.Image:
    skew_amount = 0.3
    skew_offset = int(image.size[1] * skew_amount // 2)
    
    skewed_image = image.transform(
        image.size, Image.AFFINE, (1, skew_amount, -skew_offset, 0, 1, 0), resample=Image.NEAREST
    )
    
    return skewed_image

def save_image(
    image: Image.Image,
    output_dir: str,
    case_label: str,
    letter_index: int,
    size: int,
    style: str,
) -> None:
    style_dir = os.path.join(output_dir, case_label, style)
    os.makedirs(style_dir, exist_ok=True)

    filename = f"{letter_index:02}_{size}.png"
    image.save(os.path.join(style_dir, filename))


def generate_images(
    alphabet: List[str],
    case_label: str,
    font_path: str,
    font_sizes: List[int],
    output_dir: str,
) -> None:
    for i, letter in enumerate(alphabet):
        for size in font_sizes:
            font = ImageFont.truetype(font_path, size)

            regular_image = create_image_with_text(letter, font)
            save_image(regular_image, output_dir, case_label, i + 1, size, "regular")

            bold_image = create_image_with_text(letter, font, style_func=apply_bold)
            save_image(bold_image, output_dir, case_label, i + 1, size, "bold")

            italic_image = create_image_with_text(
                letter, font, transform_func=apply_italic_transform
            )
            save_image(italic_image, output_dir, case_label, i + 1, size, "italic")


def main():
    osmanya_alphabet: List[str] = [chr(i) for i in range(0x10480, 0x104AA)]
    georgian_alphabet: List[str] = list("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")

    regular_font_path_osmanya: str = "./osmanya-regular.ttf"
    regular_font_path_georgian: str = "./georgian-regular.ttf"

    font_sizes: List[int] = [120, 150, 180]
    output_dir: str = "letter_images"

    os.makedirs(output_dir, exist_ok=True)

    generate_images(
        osmanya_alphabet,
        "osmanya",
        regular_font_path_osmanya,
        font_sizes,
        output_dir,
    )

    generate_images(
        georgian_alphabet,
        "georgian",
        regular_font_path_georgian,
        font_sizes,
        output_dir,
    )


if __name__ == "__main__":
    main()
