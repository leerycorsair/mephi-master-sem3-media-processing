import os
from tools import (
    binarize,
    filter_image,
    load_image,
    save_image,
    xor,
)


def process_single_image(input_path: str, output_dir: str, k: int, aperture_size: int):
    filename = os.path.basename(input_path)
    input_image = load_image(input_path)
    binary_image = binarize(input_image)

    output_image = filter_image(binary_image, k, aperture_size)
    output_path = os.path.join(output_dir, f"output_{filename}")
    save_image(output_image, output_path)

    xor_image = xor(input_image, output_image)
    xor_path = os.path.join(output_dir, f"xor_{filename}")
    save_image(xor_image, xor_path)


def process_images_in_folder(
    input_dir: str, output_dir: str, k: int, aperture_size: int
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".bmp")):
            input_path = os.path.join(input_dir, filename)
            process_single_image(input_path, output_dir, k, aperture_size)


def get_user_input():
    output_dir = input("Введите путь для сохранения обработанных изображений: ")
    k = int(input("Введиье коэффициент k: "))
    aperture_size = int(input("Введиье размер апертуры: "))

    return output_dir, k, aperture_size


def main():
    while True:
        print("\nМеню:")
        print("1. Обработать одно изображение")
        print("2. Обработать все изображения в папке")
        print("0. Выход")

        choice = input("Выберите пункт меню: ")

        if choice == "1":
            input_path = input("Введите путь к изображению (формат bmp или png): ")
            output_dir, k, aperture_size = get_user_input()
            process_single_image(input_path, output_dir, k, aperture_size)

        elif choice == "2":
            input_dir = input(
                "Введите путь к папке с изображениями (формат bmp или png): "
            )
            output_dir, k, aperture_size = get_user_input()
            process_images_in_folder(input_dir, output_dir, k, aperture_size)

        elif choice == "0":
            print("Выход из программы.")
            break

        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
