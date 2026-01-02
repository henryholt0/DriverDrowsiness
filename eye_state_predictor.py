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
    if eye_img is None or eye_img.size == 0:
        raise ValueError("Empty eye image provided to predictor.")

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
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_colour = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_colour, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eye_img = roi_colour[ey:ey + eh, ex:ex + ew]
                try:
                    eye_state = predict_eye_state(eye_img)
                    label = "Open" if eye_state == 1 else "Closed"
                    cv2.putText(roi_colour, label, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    print(f"Prediction error: {e}")

        cv2.imshow("Eye State Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()