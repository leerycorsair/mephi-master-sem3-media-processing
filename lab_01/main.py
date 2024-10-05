from tools import (
    compress_image,
    load_image,
    resample_image_one_pass,
    resample_image_two_pass,
    save_image,
    stretch_image,
)


def process_image(operation: str) -> None:
    image_path = input("Введите путь к изображению (формат bmp или png): ")
    image = load_image(image_path)

    if image is None:
        return

    if operation == "stretch":
        m = float(input("Введите коэффициент растяжения (M): "))
        processed_image = stretch_image(image, m)
    elif operation == "compress":
        n = float(input("Введите коэффициент сжатия (N): "))
        processed_image = compress_image(image, n)
    elif operation == "resample_two_pass":
        m = float(input("Введите коэффициент растяжения (M): "))
        n = float(input("Введите коэффициент сжатия (N): "))
        processed_image = resample_image_two_pass(image, m, n)
    elif operation == "resample_one_pass":
        k = float(input("Введите коэффициент передискретизации (K): "))
        processed_image = resample_image_one_pass(image, k)
    else:
        print("Неверная операция.")
        return

    output_path = input("Введите путь для сохранения изображения: ")
    save_image(processed_image, output_path)


def menu() -> None:
    operations = {
        "1": "stretch",
        "2": "compress",
        "3": "resample_two_pass",
        "4": "resample_one_pass",
        "0": "exit",
    }

    while True:
        print("\nМеню:")
        print("1. Растяжение изображения (интерполяция)")
        print("2. Сжатие изображения (децимация)")
        print("3. Передискретизация (растяжение, затем сжатие)")
        print("4. Передискретизация за один проход")
        print("0. Выход")

        choice = input("Выберите пункт меню: ")

        if choice in operations:
            if choice == "0":
                print("Выход из программы.")
                break
            else:
                process_image(operations[choice])
        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    menu()
