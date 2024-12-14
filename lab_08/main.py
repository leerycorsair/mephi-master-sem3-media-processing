import os

from tools import (contrast, gamma_correction, grayscale, haralick_matrix,
                   load_image, local, save_glcm_images, save_image,
                   save_metrics_to_file)


def process_image(input_image, output_dir, filename, gamma_value: float):
    gamma_image = gamma_correction(input_image, gamma_value)
    gray_image = grayscale(gamma_image)
    grayscale_path = os.path.join(output_dir, f"{filename}_grayscale_{gamma_value}.png")
    save_image(gray_image, grayscale_path)

    glcm = haralick_matrix(gray_image)
    save_glcm_images(glcm, output_dir, f"{filename}_{gamma_value}")

    metrics = {
        "Контраст (contrast)": contrast(glcm),
        "Локальная однородность (local uniformity)": local(glcm),
    }
    
    save_metrics_to_file(metrics, output_dir, f"{filename}_{gamma_value}")


def process_single_image(input_path: str, output_dir: str):
    filename = os.path.basename(input_path)
    input_image = load_image(input_path)

    input_image_path = os.path.join(output_dir, f"{filename}_input.png")
    save_image(input_image, input_image_path)

    gamma_values = [1.0, 0.5, 1.5]
    for gamma_value in gamma_values:
        process_image(input_image, output_dir, filename, gamma_value)


def process_images_in_folder(input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".bmp")):
            input_path = os.path.join(input_dir, filename)
            process_single_image(input_path, output_dir)


def get_user_input():
    output_dir = input("Введите путь для сохранения обработанных изображений: ")

    return output_dir


def main():
    while True:
        print("\nМеню:")
        print("1. Обработать одно изображение")
        print("2. Обработать все изображения в папке")
        print("0. Выход")

        choice = input("Выберите пункт меню: ")

        if choice == "1":
            input_path = input("Введите путь к изображению (формат bmp или png): ")
            output_dir = get_user_input()
            process_single_image(input_path, output_dir)

        elif choice == "2":
            input_dir = input(
                "Введите путь к папке с изображениями (формат bmp или png): "
            )
            output_dir = get_user_input()
            process_images_in_folder(input_dir, output_dir)

        elif choice == "0":
            print("Выход из программы.")
            break

        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
