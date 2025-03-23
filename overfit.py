import matplotlib.pyplot as plt

# Example training and validation loss data (you can replace these with actual data)
epochs = range(1, 21)  # 20 epochs

# Simulated loss data for training and validation (both have 20 data points)
# Normal fitting scenario: Training loss decreases and validation loss decreases and stabilizes
training_loss = [0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.58, 0.55, 0.52, 0.5,
                 0.48, 0.46, 0.44, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36]

validation_loss = [0.95, 0.92, 0.88, 0.85, 0.83, 0.8, 0.78, 0.76, 0.74, 0.73,
                   0.72, 0.7, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62]

# Plotting the graph
plt.figure(figsize=(10,6))
plt.plot(epochs, training_loss, label='Training Loss', color='blue')
plt.plot(epochs, validation_loss, label='Validation Loss', color='red')
plt.title('Normal Fitting Graph: Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
