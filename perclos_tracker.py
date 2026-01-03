import cv2
import numpy as np
from tensorflow.keras.models import load_model
from eye_state_predictor import predict_eye_state
import mediapipe as mp

# Load pre-trained eye state CNN model
model = load_model("eye_state_cnn.h5")

# Video source (0 for webcam or path to video file)
VIDEO_SOURCE = "data/test_clip_car.mp4"  # Change to 0 for webcam

# PERCLOS calculation setup
FRAME_WINDOW = 150  # ~5 seconds at 30 FPS
eye_state_history = []

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye landmark indices
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380]

cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    left_eye_img = None
    right_eye_img = None
    left_bbox = None
    right_bbox = None
    left_open = 1
    right_open = 1

    if results.multi_face_landmarks:
        h, w = frame.shape[:2]
        face_landmarks = results.multi_face_landmarks[0]
        # Extract left and right eye coordinates
        left_eye_pts = []
        right_eye_pts = []
        for idx in LEFT_EYE_IDX:
            lm = face_landmarks.landmark[idx]
            left_eye_pts.append((int(lm.x * w), int(lm.y * h)))
        for idx in RIGHT_EYE_IDX:
            lm = face_landmarks.landmark[idx]
            right_eye_pts.append((int(lm.x * w), int(lm.y * h)))
        # Bounding boxes
        lx, ly, lw_, lh_ = cv2.boundingRect(np.array(left_eye_pts))
        rx, ry, rw_, rh_ = cv2.boundingRect(np.array(right_eye_pts))
        # Add some margin
        l_margin = 5
        r_margin = 5
        lx1 = max(lx - l_margin, 0)
        ly1 = max(ly - l_margin, 0)
        lx2 = min(lx + lw_ + l_margin, w)
        ly2 = min(ly + lh_ + l_margin, h)
        rx1 = max(rx - r_margin, 0)
        ry1 = max(ry - r_margin, 0)
        rx2 = min(rx + rw_ + r_margin, w)
        ry2 = min(ry + rh_ + r_margin, h)
        left_bbox = (lx1, ly1, lx2, ly2)
        right_bbox = (rx1, ry1, rx2, ry2)
        left_eye_img = frame[ly1:ly2, lx1:lx2]
        right_eye_img = frame[ry1:ry2, rx1:rx2]
        # Only predict if region is not empty
        if left_eye_img.size > 0 and right_eye_img.size > 0:
            left_open = predict_eye_state(left_eye_img)
            right_open = predict_eye_state(right_eye_img)
            # Record whether eyes are closed (0 = closed, 1 = open)
            eyes_closed = int(left_open == 0 and right_open == 0)
            eye_state_history.append(eyes_closed)
            # Maintain history window
            if len(eye_state_history) > FRAME_WINDOW:
                eye_state_history.pop(0)
        # Draw rectangles and labels
        if left_bbox:
            cv2.rectangle(frame, (left_bbox[0], left_bbox[1]), (left_bbox[2], left_bbox[3]), (0,255,0) if left_open else (0,0,255), 2)
            cv2.putText(frame, f"Left: {'Open' if left_open else 'Closed'}", (left_bbox[0], left_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if left_open else (0,0,255), 2)
        if right_bbox:
            cv2.rectangle(frame, (right_bbox[0], right_bbox[1]), (right_bbox[2], right_bbox[3]), (0,255,0) if right_open else (0,0,255), 2)
            cv2.putText(frame, f"Right: {'Open' if right_open else 'Closed'}", (right_bbox[0], right_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if right_open else (0,0,255), 2)
    else:
        # If no face detected, do not update history to avoid accidental inflation
        pass

    # Compute PERCLOS
    perclos = sum(eye_state_history) / len(eye_state_history) if len(eye_state_history) > 0 else 0.0

    # Display info
    status = "TIRED" if perclos > 0.4 else "Awake"
    # Move PERCLOS display lower (e.g., y=100)
    cv2.putText(frame, f"PERCLOS: {perclos:.2f} - {status}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if status == "TIRED" else (0, 255, 0), 2)

    # Microsleep detection logic
    MICROSLEEP_THRESHOLD = 5  # adjust as needed
    if 'consecutive_closed_frames' not in locals():
        consecutive_closed_frames = 0

    # Use eyes_closed from above if available, else default to False
    eyes_closed_var = locals().get("eyes_closed", 0)
    if eyes_closed_var:
        consecutive_closed_frames += 1
    else:
        consecutive_closed_frames = 0

    if consecutive_closed_frames >= MICROSLEEP_THRESHOLD:
        cv2.putText(frame, "MICROSLEEP DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    cv2.imshow("PERCLOS Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
