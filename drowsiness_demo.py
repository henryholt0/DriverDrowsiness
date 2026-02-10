import numpy as np
import cv2
from eye_state_predictor import predict_eye_state
import mediapipe as mp

VIDEO_SOURCE = "data/test_clip_car.mp4"
TIRED_THRESHOLD_SECONDS = 1.5
MICROSLEEP_THRESHOLD = 5  # consecutive frames
TILT_THRESHOLD_DEGREES = 15
EYE_EAR_CLOSED_THRESH = 0.19
EYE_EAR_MIN_WIDTH = 8

# Eye landmark indices
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380]
LEFT_EYE_EAR_IDX = (33, 133, 159, 145, 158, 153)
RIGHT_EYE_EAR_IDX = (362, 263, 386, 374, 385, 380)

# Init mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(VIDEO_SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS)
CLOSED_FRAMES_THRESHOLD = int(fps * TIRED_THRESHOLD_SECONDS)
FRAME_WINDOW = int(fps * 5)

left_eye_closed_frames = 0
right_eye_closed_frames = 0
consecutive_closed_frames = 0

eye_state_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    tired_basic = False
    tired_perclos = False
    tired_tilt = False

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Eye state detection
        def compute_ear(indices):
            pts = [np.array([face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]) for i in indices]
            p1, p4, p2, p6, p3, p5 = pts
            width = np.linalg.norm(p1 - p4)
            if width < EYE_EAR_MIN_WIDTH:
                return None
            ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * width)
            return float(ear)

        def extract_eye_state(indices):
            pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]
            x_coords, y_coords = zip(*pts)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            eye_img = frame[y_min:y_max, x_min:x_max]
            if eye_img.size == 0:
                return 1, None  # Assume open if detection fails
            state = predict_eye_state(eye_img)
            return state, (x_min, y_min, x_max, y_max)

        left_state, left_box = extract_eye_state(LEFT_EYE_IDX)
        right_state, right_box = extract_eye_state(RIGHT_EYE_IDX)

        left_ear = compute_ear(LEFT_EYE_EAR_IDX)
        right_ear = compute_ear(RIGHT_EYE_EAR_IDX)
        if left_ear is not None and left_ear < EYE_EAR_CLOSED_THRESH:
            left_state = 0
        if right_ear is not None and right_ear < EYE_EAR_CLOSED_THRESH:
            right_state = 0

        # Draw eye boxes and labels
        if left_box:
            color = (0, 255, 0) if left_state else (0, 0, 255)
            cv2.putText(frame, f"Left: {'Open' if left_state else 'Closed'}", (left_box[0], left_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if right_box:
            color = (0, 255, 0) if right_state else (0, 0, 255)
            cv2.putText(frame, f"Right: {'Open' if right_state else 'Closed'}", (right_box[0], right_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # TIRED BASIC
        if left_state == 0:
            left_eye_closed_frames += 1
        else:
            left_eye_closed_frames = 0
        if right_state == 0:
            right_eye_closed_frames += 1
        else:
            right_eye_closed_frames = 0

        if (left_eye_closed_frames >= CLOSED_FRAMES_THRESHOLD and
                right_eye_closed_frames >= CLOSED_FRAMES_THRESHOLD):
            tired_basic = True
            cv2.putText(frame, "TIRED (Closed Eyes)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # PERCLOS
        both_closed = int(left_state == 0 and right_state == 0)
        eye_state_history.append(both_closed)
        if len(eye_state_history) > FRAME_WINDOW:
            eye_state_history.pop(0)
        perclos = sum(eye_state_history) / len(eye_state_history)
        if perclos > 0.4:
            tired_perclos = True
        color = (0, 0, 255) if tired_perclos else (0, 255, 0)
        cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        # MICROSLEEP
        if both_closed:
            consecutive_closed_frames += 1
        else:
            consecutive_closed_frames = 0

        if consecutive_closed_frames >= MICROSLEEP_THRESHOLD:
            cv2.putText(frame, "MICROSLEEP DETECTED", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # HEAD TILT
        p1 = (int(face_landmarks.landmark[33].x * w), int(face_landmarks.landmark[33].y * h))
        p2 = (int(face_landmarks.landmark[263].x * w), int(face_landmarks.landmark[263].y * h))
        tilt_angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        cv2.putText(frame, f"Tilt: {tilt_angle:.1f} deg", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
        if abs(tilt_angle) > TILT_THRESHOLD_DEGREES:
            tired_tilt = True
            cv2.putText(frame, "HEAD TILT DETECTED", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Driver Drowsiness Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
