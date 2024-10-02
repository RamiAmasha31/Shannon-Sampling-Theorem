import numpy as np
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt

# Load the audio file
filename = 'speech1.wav'  # Replace with the path to your sound file
signal, sr = librosa.load(filename, sr=None)

# Set the sampling parameters
T = 1 / sr  # Sampling period
Fs = sr  # Sampling frequency

# Define the frequency range for bandlimiting
low_freq = 1000  # Lower frequency limit (in Hz)
high_freq = 3000  # Upper frequency limit (in Hz)

# Generate the sampled points
sample_points = np.arange(0, len(signal) * T, T)
sample_values = signal

# Apply windowing to the sampled values for bandlimiting
window = np.hanning(len(sample_values))
windowed_values = sample_values * window

# Calculate the Fourier Transform of the windowed values
fft_windowed = np.fft.fft(windowed_values)
freq_windowed = np.fft.fftfreq(len(sample_points), d=T)

# Zero out frequencies outside the desired range
fft_windowed[(freq_windowed < low_freq) | (freq_windowed > high_freq)] = 0

# Reconstruct the signal in the time domain
reconstructed_points = np.linspace(0, len(signal) * T, len(signal))
reconstructed_signal = np.zeros_like(reconstructed_points)

for i in range(len(sample_points)):
    reconstructed_signal += sample_values[i] * np.sinc((reconstructed_points - sample_points[i]) / T)

# Reconstruct the signal in the frequency domain
reconstructed_freq_domain = np.fft.ifft(fft_windowed).real

# Normalize the reconstructed signal
reconstructed_freq_domain /= np.max(np.abs(reconstructed_freq_domain))

# Play the original signal
sd.play(signal, sr)
sd.wait()

# Play the reconstructed signal in the frequency domain
sd.play(reconstructed_freq_domain, sr)
sd.wait()

# Plotting
plt.figure(figsize=(12, 8))

# Plot the original signal
plt.subplot(2, 2, 1)
time = np.arange(len(signal)) / sr
plt.plot(time, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal')

# Plot the sampled points
plt.subplot(2, 2, 2)
plt.stem(sample_points, sample_values, 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sampled Signal')

# Plot the Fourier Transform of the windowed signal (bandlimited)
plt.subplot(2, 2, 3)
plt.stem(freq_windowed, np.abs(fft_windowed))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Bandlimited Fourier Transform')

# Plot the reconstructed signal in the frequency domain
plt.subplot(2, 2, 4)
plt.plot(reconstructed_points, reconstructed_freq_domain)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal (Frequency Domain)')

plt.tight_layout()
plt.show()
