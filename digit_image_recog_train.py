import numpy as np
import pandas as pd
import os

# Sigmoid activation function
def sigmoid(x):
    x = np.clip(x, -500, 500)  # prevent overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Softmax activation function for output layer
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # numerical stability
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# =========================
# Load dataset from CSV
# =========================

csv_path = r"Digit Model v2\mnist_test.csv"

df = pd.read_csv(csv_path)

# First column = label, remaining columns = pixels
y = df.iloc[:, 0].values.astype(int)
X = df.iloc[:, 1:].values.astype(np.float32)

# Normalize pixel values from 0-255 to 0-1
X = X / 255.0

# Validate input size
if X.shape[1] != 784:
    raise ValueError(f"Expected 784 pixel columns, but got {X.shape[1]} columns.")

# One-hot encode labels
y_onehot = np.zeros((y.size, 10))
y_onehot[np.arange(y.size), y] = 1

# =========================
# Neural network setup
# =========================

np.random.seed(1)

weights_path = r"C:\Figma\Digit Model\nn_weights.npz"
os.makedirs(os.path.dirname(weights_path), exist_ok=True)

# Load previous weights or initialize
if os.path.exists(weights_path):
    weights = np.load(weights_path)

    weights_input_hidden1 = weights["weights_input_hidden1"]
    weights_hidden1_hidden2 = weights["weights_hidden1_hidden2"]
    weights_hidden2_output = weights["weights_hidden2_output"]

    # Reinitialize if shape does not match
    if weights_input_hidden1.shape != (784, 128):
        weights_input_hidden1 = 2 * np.random.random((784, 128)) - 1

    if weights_hidden1_hidden2.shape != (128, 64):
        weights_hidden1_hidden2 = 2 * np.random.random((128, 64)) - 1

    if weights_hidden2_output.shape != (64, 10):
        weights_hidden2_output = 2 * np.random.random((64, 10)) - 1

else:
    weights_input_hidden1 = 2 * np.random.random((784, 128)) - 1
    weights_hidden1_hidden2 = 2 * np.random.random((128, 64)) - 1
    weights_hidden2_output = 2 * np.random.random((64, 10)) - 1

# Learning rate and epochs
lr = 0.1
epochs = 5000
n_samples = X.shape[0]

# =========================
# Training loop
# =========================

for epoch in range(epochs):
    # Forward propagation
    hidden1_output = sigmoid(np.dot(X, weights_input_hidden1))
    hidden2_output = sigmoid(np.dot(hidden1_output, weights_hidden1_hidden2))

    final_input = np.dot(hidden2_output, weights_hidden2_output)
    final_output = softmax(final_input)

    # Error
    error = y_onehot - final_output

    # Backpropagation
    # Dividing by n_samples makes training more stable for CSV datasets
    d_final = error / n_samples

    d_hidden2 = np.dot(d_final, weights_hidden2_output.T) * sigmoid_derivative(hidden2_output)
    d_hidden1 = np.dot(d_hidden2, weights_hidden1_hidden2.T) * sigmoid_derivative(hidden1_output)

    # Update weights
    weights_hidden2_output += np.dot(hidden2_output.T, d_final) * lr
    weights_hidden1_hidden2 += np.dot(hidden1_output.T, d_hidden2) * lr
    weights_input_hidden1 += np.dot(X.T, d_hidden1) * lr

    if epoch % 500 == 0:
        loss = -np.mean(np.sum(y_onehot * np.log(final_output + 1e-8), axis=1))
        predictions = np.argmax(final_output, axis=1)
        accuracy = np.mean(predictions == y) * 100

        print(f"Epoch {epoch}, loss={loss:.4f}, accuracy={accuracy:.2f}%")

# =========================
# Save weights
# =========================

np.savez(
    weights_path,
    weights_input_hidden1=weights_input_hidden1,
    weights_hidden1_hidden2=weights_hidden1_hidden2,
    weights_hidden2_output=weights_hidden2_output
)

print("Training complete and weights saved with softmax output.")
