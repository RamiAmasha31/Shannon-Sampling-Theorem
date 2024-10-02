import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import resample
import librosa
# Load the audio file
filename = 'speech1.wav'  # Replace with the path to your sound file
signal, sr = librosa.load(filename, sr=None)

# Set the desired sampling rate (lower than the original sr)
desired_sr = 1000  # Replace with your desired sampling rate (lower than the original sr)

# Resample the signal to the desired sampling rate
signal_resampled = resample(signal, int(len(signal) * desired_sr / sr))

# Update the sampling rate
sr = desired_sr

# Set the sampling parameters
T = 1 / sr  # Sampling period

# Generate the sampled points
sample_points = np.arange(0, len(signal_resampled) * T, T)
sample_values = signal_resampled

# Reconstruct the signal using Shannon sampling theorem
reconstructed_points = np.linspace(0, len(signal_resampled) * T, len(signal_resampled))
reconstructed_signal = np.zeros_like(reconstructed_points)

for i in range(len(sample_points)):
    reconstructed_signal += sample_values[i] * np.sinc((reconstructed_points - sample_points[i]) / T)

# Play the original signal
filename = 'speech1.wav'  # Replace with the path to your sound file
signal1, sr1 = librosa.load(filename, sr=None)
sd.play(signal1, sr1)
sd.wait()

# Play the reconstructed signal
sd.play(reconstructed_signal, sr)
sd.wait()




# Plotting
plt.figure(figsize=(12, 8))

# Plot the original signal
plt.subplot(3, 1, 1)
time = np.arange(len(signal)) / sr
plt.plot(time, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal')

# Plot the sampled signal
plt.subplot(3, 1, 2)
time_sampled = np.arange(len(signal_resampled)) / sr
plt.plot(time_sampled, signal_resampled)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sampled Signal')

# Plot the reconstructed signal
plt.subplot(3, 1, 3)
plt.plot(reconstructed_points, reconstructed_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal')

plt.tight_layout()
plt.show()
