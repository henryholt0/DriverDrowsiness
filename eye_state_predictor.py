import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load the trained CNN model (make sure this .h5 file exists in your project folder)
model = load_model('eye_state_cnn.h5')

def predict_eye_state(eye_img):
    # Resize to 90x90 (as expected by the model)
    eye_resized = cv2.resize(eye_img, (90, 90))

    # Normalize pixel values to [0, 1]
    eye_normalized = eye_resized / 255.0

    # Convert to float32 and reshape to (1, 90, 90, 3)
    eye_input = np.expand_dims(eye_normalized.astype('float32'), axis=0)

    # Predict using the model
    prediction = model.predict(eye_input)

    # Return 0 (closed) or 1 (open) based on highest probability
    return int(np.argmax(prediction))

if __name__ == "__main__":
    import os

    # Load and test on a single image
    test_img_path = "test_eye.png"  # put a sample eye image in your folder
    if os.path.exists(test_img_path):
        test_img = cv2.imread(test_img_path)
        result = predict_eye_state(test_img)
        print("Prediction:", "Open" if result == 1 else "Closed")
    else:
        print("No test_eye.png found")
