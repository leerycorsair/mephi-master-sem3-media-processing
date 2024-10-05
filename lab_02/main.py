import os
from tools import load_image, niblack_binarization, rgb_to_grayscale, save_image


def process_single_image(input_path: str, output_dir: str, window_size: int, k: float):
    filename = os.path.basename(input_path)
    image = load_image(input_path)

    greyscale_image = rgb_to_grayscale(image)
    grayscale_output_path = os.path.join(output_dir, f"gray_{filename}")
    save_image(greyscale_image, grayscale_output_path)

    binary_image = niblack_binarization(greyscale_image, window_size, k)
    binary_output_path = os.path.join(output_dir, f"binary_{filename}")
    save_image(binary_image, binary_output_path)


def process_images_in_folder(
    input_dir: str, output_dir: str, window_size: int, k: float
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".bmp")):
            input_path = os.path.join(input_dir, filename)
            process_single_image(input_path, output_dir, window_size, k)


def get_user_input():
    window_size = int(
        input("Введите размер окна для локальной окрестности (например, 15): ")
    )
    k = float(input("Введите параметр k (например, -0.2): "))
    output_dir = input("Введите путь для сохранения обработанных изображений: ")
    return window_size, k, output_dir


def main():
    while True:
        print("\nМеню:")
        print("1. Обработать одно изображение")
        print("2. Обработать все изображения в папке")
        print("0. Выход")

        choice = input("Выберите пункт меню: ")

        if choice == "1":
            input_path = input("Введите путь к изображению (формат bmp или png): ")
            window_size, k, output_dir = get_user_input()
            process_single_image(input_path, output_dir, window_size, k)

        elif choice == "2":
            input_dir = input(
                "Введите путь к папке с изображениями (формат bmp или png): "
            )
            window_size, k, output_dir = get_user_input()
            process_images_in_folder(input_dir, output_dir, window_size, k)

        elif choice == "0":
            print("Выход из программы.")
            break

        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
