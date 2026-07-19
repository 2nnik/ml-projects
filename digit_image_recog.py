import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# =========================
# Activation functions
# =========================

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# =========================
# Load trained weights
# =========================

weights_path = r"C:\Figma\Digit Model\nn_weights.npz"

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weight file not found: {weights_path}")

weights = np.load(weights_path)

weights_input_hidden1 = weights["weights_input_hidden1"]
weights_hidden1_hidden2 = weights["weights_hidden1_hidden2"]
weights_hidden2_output = weights["weights_hidden2_output"]

# =========================
# Prediction function
# =========================

def predict_digit(img_array_28x28):
    """
    img_array_28x28 must be a 28x28 grayscale image.
    Pixel value 0 = black background.
    Pixel value 255 = white digit stroke.
    """

    X_test = img_array_28x28.flatten().reshape(1, -1).astype(np.float32) / 255.0

    hidden1_output = sigmoid(np.dot(X_test, weights_input_hidden1))
    hidden2_output = sigmoid(np.dot(hidden1_output, weights_hidden1_hidden2))

    final_input = np.dot(hidden2_output, weights_hidden2_output)
    final_output = softmax(final_input)

    predicted_digit = np.argmax(final_output)
    confidence = np.max(final_output)

    return predicted_digit, confidence, final_output

# =========================
# Drawing canvas settings
# =========================

canvas_size = 280
mnist_size = 28
brush_size = 18

# Black background, white digit
canvas = Image.new("L", (canvas_size, canvas_size), color=0)
draw = ImageDraw.Draw(canvas)

last_position = None

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_title("Draw a digit, then close the window")
ax.imshow(canvas, cmap="gray", vmin=0, vmax=255)
ax.axis("off")

def update_display():
    ax.clear()
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Draw a digit, then close the window")
    ax.axis("off")
    fig.canvas.draw_idle()

def on_press(event):
    global last_position

    if event.xdata is None or event.ydata is None:
        return

    x, y = int(event.xdata), int(event.ydata)
    last_position = (x, y)

def on_motion(event):
    global last_position

    if event.xdata is None or event.ydata is None:
        return

    if event.button != 1:
        return

    x, y = int(event.xdata), int(event.ydata)

    if last_position is not None:
        draw.line(
            [last_position, (x, y)],
            fill=255,
            width=brush_size,
            joint="curve"
        )

    last_position = (x, y)
    update_display()

def on_release(event):
    global last_position
    last_position = None

def on_key(event):
    global canvas, draw

    if event.key == "c":
        canvas = Image.new("L", (canvas_size, canvas_size), color=0)
        draw = ImageDraw.Draw(canvas)
        update_display()
        print("Canvas cleared.")

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("key_press_event", on_key)

print("Draw using the left mouse button.")
print("Press 'c' to clear the canvas.")
print("Close the window when finished.")

plt.show()

# =========================
# Convert drawing to MNIST format
# =========================

img_28 = canvas.resize((mnist_size, mnist_size), Image.Resampling.LANCZOS)
img_array = np.array(img_28, dtype=np.uint8)

# Optional thresholding to make the digit clearer
img_array[img_array < 30] = 0

# =========================
# Predict digit
# =========================

pred_digit, confidence, probabilities = predict_digit(img_array)

print("\nPrediction result")
print("=================")
print(f"Predicted digit : {pred_digit}")
print(f"Confidence      : {confidence * 100:.2f}%")
print("Probabilities   :")

for digit, prob in enumerate(probabilities[0]):
    print(f"{digit}: {prob:.4f}")

# =========================
# Show final 28x28 image
# =========================

plt.figure(figsize=(3, 3))
plt.imshow(img_array, cmap="gray", vmin=0, vmax=255)
plt.title(f"Predicted digit: {pred_digit}")
plt.axis("off")
plt.show()

# =========================
# Optional: save drawn test image
# =========================

save_choice = input("Save this drawn digit? (y/n): ").strip().lower()

if save_choice == "y":
    label = input("Enter the correct digit label (0-9): ").strip()

    if label.isdigit() and 0 <= int(label) <= 9:
        save_dir = r"C:\Figma\Digit Model\drawn_test_digits"
        os.makedirs(save_dir, exist_ok=True)

        existing_files = [
            f for f in os.listdir(save_dir)
            if f.startswith(f"{label}_drawn_") and f.endswith(".png")
        ]

        index = len(existing_files) + 1
        filename = f"{label}_drawn_{index}.png"

        Image.fromarray(img_array).save(os.path.join(save_dir, filename))

        print(f"Saved as: {filename}")
    else:
        print("Invalid label. Image was not saved.")
