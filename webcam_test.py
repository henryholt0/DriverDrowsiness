import numpy as np
import cv2
from eye_state_predictor import predict_eye_state

try:
    import mediapipe as mp
except Exception as exc:
    raise ImportError(
        "mediapipe is required for face landmarks. "
        "Install it with `pip install mediapipe` (or `conda install -c conda-forge mediapipe`)."
    ) from exc

if not hasattr(mp, "solutions"):
    version = getattr(mp, "__version__", "unknown")
    raise ImportError(
        "Installed 'mediapipe' package is missing `solutions` "
        f"(version: {version}). This script expects the older Solutions API. "
        "Install a compatible build like `pip install mediapipe==0.10.14`."
    )

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
VIDEO_SOURCE = "data/test_clip.mp4"  # Change to 0 for webcam, or provide path to mp4
cap = cv2.VideoCapture(VIDEO_SOURCE)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30  # fallback if FPS cannot be read

TIRED_THRESHOLD_SECONDS = 1.5
CLOSED_FRAMES_THRESHOLD = int(fps * TIRED_THRESHOLD_SECONDS)

left_eye_closed_frames = 0
right_eye_closed_frames = 0

if not cap.isOpened():
    print(f"⚠️ Could not open video source: {VIDEO_SOURCE}")
    exit()

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Left eye landmarks (around eye region)
            left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153]
            left_eye_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
            x_coords, y_coords = zip(*left_eye_points)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            left_eye_img = frame[y_min:y_max, x_min:x_max]
            if left_eye_img.size != 0:
                eye_state = predict_eye_state(left_eye_img)
                label = "Open" if eye_state == 1 else "Closed"

                if eye_state == 0:
                    left_eye_closed_frames += 1
                else:
                    left_eye_closed_frames = 0

                cv2.putText(frame, f"Left: {label}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

            # Right eye landmarks (around eye region)
            right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380]
            right_eye_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]
            x_coords, y_coords = zip(*right_eye_points)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            right_eye_img = frame[y_min:y_max, x_min:x_max]
            if right_eye_img.size != 0:
                eye_state = predict_eye_state(right_eye_img)
                label = "Open" if eye_state == 1 else "Closed"

                if eye_state == 0:
                    right_eye_closed_frames += 1
                else:
                    right_eye_closed_frames = 0

                cv2.putText(frame, f"Right: {label}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    if (left_eye_closed_frames >= CLOSED_FRAMES_THRESHOLD and
            right_eye_closed_frames >= CLOSED_FRAMES_THRESHOLD):
        cv2.putText(frame, "TIRED", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
