import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "eye_state_cnn.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. Train the model with eye_state_cnn.py first."
    )

# Load the trained CNN model
model = load_model(MODEL_PATH)


def _prepare_eye_input(eye_img):
    """
    Ensure the eye image is in the shape and scale expected by the model:
    - RGB channels
    - 90x90 resolution
    - float32 values in [0, 1]
    Returns an array shaped (1, 90, 90, 3).
    """
    if eye_img is None or eye_img.size == 0:
        raise ValueError("Empty eye image provided to predictor.")

    # Convert grayscale/single-channel images to RGB; convert BGR to RGB for consistency
    if len(eye_img.shape) == 2 or eye_img.shape[2] == 1:
        eye_rgb = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2RGB)
    else:
        eye_rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)

    eye_resized = cv2.resize(eye_rgb, (90, 90))
    eye_normalized = (eye_resized / 255.0).astype("float32")
    return np.expand_dims(eye_normalized, axis=0)

def predict_eye_state(eye_img):
    eye_input = _prepare_eye_input(eye_img)
    prediction = model.predict(eye_input)
    prob_open = float(prediction[0][0])
    return int(prob_open >= 0.5)

if __name__ == "__main__":
    # Load and test on a single image
    test_img_path = "test_eye.png"  # put a sample eye image in your folder
    if os.path.exists(test_img_path):
        test_img = cv2.imread(test_img_path)
        result = predict_eye_state(test_img)
        print("Prediction:", "Open" if result == 1 else "Closed")
    else:
        print("No test_eye.png found")
