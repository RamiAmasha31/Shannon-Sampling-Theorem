# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:25:41 2023

@author: ראמי
"""


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

# Generate the sampled points
sample_points = np.arange(0, len(signal) * T, T)
sample_values = signal

# Frequency spectrum of the sampled signal
fft_samples = np.fft.fft(sample_values)
freq = np.fft.fftfreq(len(sample_points), d=T)

# Reconstruct the signal
reconstructed_points = np.linspace(0, len(signal) * T, len(signal))
reconstructed_signal = np.zeros_like(reconstructed_points)

for i in range(len(sample_points)):
    reconstructed_signal += sample_values[i] * np.sinc((reconstructed_points - sample_points[i]) / T)

# Play the original signal
sd.play(signal, sr)

# Wait for the original signal to finish playing
sd.wait()

# Play the reconstructed signal
sd.play(reconstructed_signal, sr)

# Wait for the reconstructed signal to finish playing
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

# Plot the frequency spectrum
plt.subplot(2, 2, 3)
plt.stem(freq, np.abs(fft_samples))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')

# Plot the reconstructed signal
plt.subplot(2, 2, 4)
plt.plot(reconstructed_points, reconstructed_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal')

plt.tight_layout()
plt.show()
