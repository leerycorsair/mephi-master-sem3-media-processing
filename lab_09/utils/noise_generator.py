import numpy as np
import soundfile as sf
from scipy.signal import lfilter
import matplotlib.pyplot as plt


def generate_white_noise(length):
    return np.random.normal(0, 1, length)


def generate_uniform_noise(length):
    return np.random.uniform(-1, 1, length)


def generate_pink_noise(length):
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    white = np.random.normal(0, 1, length)
    pink = lfilter(b, a, white)
    return pink


def generate_brown_noise(length):
    brown = np.cumsum(np.random.normal(0, 1, length))
    brown = brown / np.max(np.abs(brown))
    return brown


def add_random_noise_to_wav(
    input_file,
    output_file,
    snr_db=20,
    noise_types=None,
    combine_ratio=0.5,
):
    if noise_types is None:
        noise_types = ["white", "pink", "brown"]
    y, sr = sf.read(input_file)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    length = len(y)
    combined_noise = np.zeros(length)
    for noise_type in noise_types:
        if noise_type == "white":
            noise = generate_white_noise(length)
        elif noise_type == "uniform":
            noise = generate_uniform_noise(length)
        elif noise_type == "pink":
            noise = generate_pink_noise(length)
        elif noise_type == "brown":
            noise = generate_brown_noise(length)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        combined_noise += noise * combine_ratio
    combined_noise = combined_noise / len(noise_types)
    signal_power = np.mean(y**2)
    noise_power = np.mean(combined_noise**2)
    snr_linear = 10 ** (snr_db / 10)
    desired_noise_power = signal_power / snr_linear
    scaling_factor = np.sqrt(desired_noise_power / noise_power)
    noise_scaled = combined_noise * scaling_factor
    y_noisy = y + noise_scaled
    max_val = np.max(np.abs(y_noisy))
    if max_val > 0:
        y_noisy = y_noisy / max_val * 0.99
    sf.write(output_file, y_noisy, sr)
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    time = np.linspace(0, len(y) / sr, num=len(y))
    plt.plot(time, y, color="blue")
    plt.title("Исходный сигнал")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.subplot(2, 1, 2)
    time_noisy = np.linspace(0, len(y_noisy) / sr, num=len(y_noisy))
    plt.plot(time_noisy, y_noisy, color="red")
    plt.title(f"Зашумленный сигнал (SNR = {snr_db} дБ)")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.tight_layout()
    plt.savefig("./utils/signals_comparison_random_noise.png")


def main():
    input_wav = "./utils/drums.wav"
    output_wav = "./utils/noisy_drums.wav"
    snr = 20
    noise_types = ["white", "uniform", "pink", "brown"]
    combine_ratio = 0.25
    add_random_noise_to_wav(
        input_wav,
        output_wav,
        snr_db=snr,
        noise_types=noise_types,
        combine_ratio=combine_ratio,
    )


if __name__ == "__main__":
    main()
