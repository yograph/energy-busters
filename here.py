import numpy as np
import matplotlib.pyplot as plt

# Define amplitudes
A_high = 5
A_low = 2

# Define the pattern
high_duration = 5
low_duration = 2

# Create the signal
signal_pattern = [A_high] * high_duration + [A_low] * low_duration
repeats = 10  # Number of times the pattern repeats
signal = signal_pattern * repeats

# Generate time points for plotting
time = np.arange(len(signal))

# Plot the signal
plt.plot(time, signal, drawstyle='steps-pre', label='Signal')
plt.title('Signal with 5 High Amplitude and 2 Low Amplitude')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()
plt.show()
