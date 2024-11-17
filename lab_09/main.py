import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, lfilter, savgol_filter
from scipy.signal.windows import hann

def plot_spectrogram(filename, output_image):
    fs, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)

    nperseg = 1024
    noverlap = nperseg // 2
    window = hann(nperseg)

    f, t, Sxx = spectrogram(data, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')

    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
    plt.yscale('log')
    plt.colorbar(label='Уровень (дБ)')
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')
    plt.title('Спектрограмма')
    plt.ylim([20, fs / 2])
    plt.savefig(output_image)
    plt.close()

def apply_savgol_filter(input_file, output_file, window_length=51, polyorder=3):
    fs, data = wavfile.read(input_file)
    if data.ndim > 1:
        data = data.mean(axis=1)
    filtered_data = savgol_filter(data, window_length, polyorder)
    filtered_data = np.int16(filtered_data / np.max(np.abs(filtered_data)) * 32767)
    wavfile.write(output_file, fs, filtered_data)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(input_file, output_file, cutoff=1000, order=6):
    fs, data = wavfile.read(input_file)
    if data.ndim > 1:
        data = data.mean(axis=1)
    b, a = butter_lowpass(cutoff, fs, order)
    filtered_data = lfilter(b, a, data)
    filtered_data = np.int16(filtered_data / np.max(np.abs(filtered_data)) * 32767)
    wavfile.write(output_file, fs, filtered_data)

def find_high_energy_moments(filename, delta_t=0.1, delta_f=(40, 50)):
    fs, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)
    nperseg = 1024
    noverlap = nperseg // 2
    window = hann(nperseg)
    f, t, Sxx = spectrogram(data, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')
    freq_mask = (f >= delta_f[0]) & (f <= delta_f[1])
    Sxx_filtered = Sxx[freq_mask, :]
    energy = Sxx_filtered.mean(axis=0)
    energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    threshold = 0.8
    high_energy_indices = np.where(energy_normalized >= threshold)[0]
    high_energy_times = []
    last_time = -np.inf
    for idx in high_energy_indices:
        current_time = t[idx]
        if current_time - last_time >= delta_t:
            high_energy_times.append(current_time)
            last_time = current_time
    return high_energy_times

def plot_spectrogram_with_moments(filename, output_image, moments):
    fs, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)
    nperseg = 1024
    noverlap = nperseg // 2
    window = hann(nperseg)
    f, t, Sxx = spectrogram(data, fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
    plt.yscale('log')
    plt.colorbar(label='Уровень (дБ)')
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')
    plt.title('Спектрограмма с моментами высокой энергии')
    plt.ylim([20, fs / 2])
    for moment in moments:
        plt.axvline(x=moment, color='cyan', linestyle='--', linewidth=1)
    plt.savefig(output_image)
    plt.close()

def main():
    original = 'noisy_drums.wav'
    savgol_file = 'noisy_drums_savgol.wav'
    lowpass_file = 'noisy_drums_lowpass.wav'
    
    plot_spectrogram(original, 'spectrogram_original.png')
    
    apply_savgol_filter(original, savgol_file)
    plot_spectrogram(savgol_file, 'spectrogram_savgol.png')
    
    apply_lowpass_filter(original, lowpass_file, cutoff=1000, order=6)
    plot_spectrogram(lowpass_file, 'spectrogram_lowpass.png')
    
    high_energy_moments = find_high_energy_moments(savgol_file, delta_t=0.1, delta_f=(40, 50))
    plot_spectrogram_with_moments(savgol_file, 'spectrogram_with_moments.png', high_energy_moments)
    
    print("Моменты времени с наибольшей энергией в диапазоне 40-50 Гц:")
    for moment in high_energy_moments:
        print(f"{moment:.2f} секунд")

if __name__ == '__main__':
    main()
