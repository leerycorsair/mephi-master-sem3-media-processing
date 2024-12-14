import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, spectrogram
from scipy.signal.windows import hann


def plot_spectrogram(filename, output_image):
    fs, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)

    nperseg = 1024
    noverlap = nperseg // 2
    window = hann(nperseg)

    f, t, Sxx = spectrogram(
        data,
        fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="magnitude",
    )

    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
    plt.yscale("log")
    plt.colorbar(label="Уровень (дБ)")
    plt.ylabel("Частота [Гц]")
    plt.xlabel("Время [с]")
    plt.title("Спектрограмма")
    plt.ylim([20, fs / 2])
    plt.savefig(output_image)
    plt.close()


def find_min_max_frequency(filename):
    fs, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)
    nperseg = 1024
    noverlap = nperseg // 2
    window = hann(nperseg)
    f, t, Sxx = spectrogram(
        data,
        fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="magnitude",
    )
    power = np.mean(Sxx, axis=1)
    nonzero_indices = np.where(power > 0)[0]
    if len(nonzero_indices) == 0:
        min_freq = 0
        max_freq = 0
    else:
        min_freq = f[nonzero_indices.min()]
        max_freq = f[nonzero_indices.max()]
    return min_freq, max_freq


def find_fundamental_tone(filename):
    fs, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)
    windowed = data * hann(len(data))
    fft_spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), 1 / fs)
    magnitude = np.abs(fft_spectrum)

    magnitude[: int(50 / (fs / len(windowed)))] = 0

    fundamental_freq = freqs[np.argmax(magnitude)]
    return fundamental_freq


def find_formants(filename, num_formants=3):
    fs, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)
    nperseg = 2048
    noverlap = nperseg // 2
    window = hann(nperseg)
    f, t, Sxx = spectrogram(
        data,
        fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="magnitude",
    )

    formants = []
    for i in range(len(t)):
        spectrum = Sxx[:, i]
        formant_mask = (f > 300) & (f < 5000)
        peaks, properties = find_peaks(
            spectrum[formant_mask],
            height=np.max(spectrum[formant_mask]) * 0.3,
            distance=40,
        )
        peak_freqs = f[formant_mask][peaks]
        if len(peak_freqs) >= num_formants:
            top_peaks = peak_freqs[
                np.argsort(properties["peak_heights"])[::-1][:num_formants]
            ]
            formants.append(sorted(top_peaks))

    formants = [f_set for f_set in formants if len(f_set) == num_formants]
    if not formants:
        return []

    formants = np.array(formants)
    formant_means = np.median(formants, axis=0)
    return formant_means


def plot_spectrogram_with_moments(filename, output_image, moments):
    fs, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)
    nperseg = 1024
    noverlap = nperseg // 2
    window = hann(nperseg)
    f, t, Sxx = spectrogram(
        data,
        fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="magnitude",
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
    plt.yscale("log")
    plt.colorbar(label="Уровень (дБ)")
    plt.ylabel("Частота [Гц]")
    plt.xlabel("Время [с]")
    plt.title("Спектрограмма с моментами высокой энергии")
    plt.ylim([20, fs / 2])
    for moment in moments:
        plt.axvline(x=moment, color="cyan", linestyle="--", linewidth=1)
    plt.savefig(output_image)
    plt.close()


def find_high_energy_moments(filename, delta_t=0.1, energy_threshold=0.8):
    fs, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)
    nperseg = 1024
    noverlap = nperseg // 2
    window = hann(nperseg)
    f, t, Sxx = spectrogram(
        data,
        fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="magnitude",
    )
    power = np.mean(Sxx, axis=0)
    energy_normalized = (power - np.min(power)) / (np.max(power) - np.min(power))
    high_energy_indices = np.where(energy_normalized >= energy_threshold)[0]

    high_energy_times = []
    last_time = -np.inf
    for idx in high_energy_indices:
        current_time = t[idx]
        if current_time - last_time >= delta_t:
            high_energy_times.append(current_time)
            last_time = current_time
    return high_energy_times


def main():
    sounds = {
        "barking": "barking.wav",
        'voice_a': 'voice_a.wav',
        'voice_e': 'voice_e.wav',  
    }

    for label, file in sounds.items():
        if not os.path.isfile(file):
            continue

        spectrogram_image = f"spectrogram_{label}.png"
        plot_spectrogram(file, spectrogram_image)

        min_freq, max_freq = find_min_max_frequency(file)
        print(f"Минимальная частота: {min_freq:.2f} Гц")
        print(f"Максимальная частота: {max_freq:.2f} Гц")

        fundamental_freq = find_fundamental_tone(file)
        print(f"Основная частота: {fundamental_freq:.2f} Гц")

        moments = find_high_energy_moments(file, delta_t=0.1, energy_threshold=0.8)

        formants = find_formants(file, num_formants=3)
        if formants.size == 0:
            print("Форманты не найдены.")
        else:
            print("Форманты:")
            for i, f_val in enumerate(formants, 1):
                print(f"  F{i}: {f_val:.2f} Гц")

        spectrogram_moments_image = f"spectrogram_{label}_with_moments.png"
        plot_spectrogram_with_moments(file, spectrogram_moments_image, moments)
        print(f"Спектрограмма с моментами сохранена как {spectrogram_moments_image}")

        print("-" * 50)


if __name__ == "__main__":
    main()
