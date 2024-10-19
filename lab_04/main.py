import os
from tools import (
    apply_scharr,
    load_image,
    save_image,
    to_binary,
    to_grayscale,
)

from dataclasses import dataclass


@dataclass
class Params:
    output_dir: str


def get_user_input():
    output_dir = input("Введите путь для сохранения обработанных изображений: ")
    return Params(
        output_dir,
    )


def save_processed_image(image, filename_suffix: str, input_path: str, params: Params):
    filename, ext = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(params.output_dir, f"{filename}_{filename_suffix}{ext}")
    save_image(image, output_path)


def process_single_image(input_path: str, params: Params):
    input_image = load_image(input_path)
    save_processed_image(input_image, "input", input_path, params)

    grayscale_image = to_grayscale(input_image)
    save_processed_image(grayscale_image, "grayscale", input_path, params)

    gx, gy, g = apply_scharr(grayscale_image)
    save_processed_image(gx, "gx", input_path, params)
    save_processed_image(gy, "gy", input_path, params)
    save_processed_image(g, "g", input_path, params)

    grayscale_g = to_binary(g, threshold=100)
    save_processed_image(grayscale_g, "grayscale_g", input_path, params)


def process_images_in_folder(input_dir: str, params: Params):
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".bmp")):
            input_path = os.path.join(input_dir, filename)
            process_single_image(input_path, params)


def main():
    while True:
        print("\nМеню:")
        print("1. Обработать одно изображение")
        print("2. Обработать все изображения в папке")
        print("0. Выход")

        choice = input("Выберите пункт меню: ")

        if choice == "1":
            input_path = input("Введите путь к изображению (формат bmp или png): ")
            params = get_user_input()
            process_single_image(input_path, params)

        elif choice == "2":
            input_dir = input(
                "Введите путь к папке с изображениями (формат bmp или png): "
            )
            params = get_user_input()
            process_images_in_folder(input_dir, params)

        elif choice == "0":
            print("Выход из программы.")
            break

        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
