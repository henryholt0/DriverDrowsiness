import cv2
import numpy as np
import mediapipe as mp

VIDEO_SOURCE = "data/test_clip_car.mp4"  # Use 0 for webcam
TILT_THRESHOLD_DEGREES = 15  # Adjust based on sensitivity

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(VIDEO_SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 33

def calculate_angle(p1, p2):
    """Returns angle in degrees between the horizontal and the line p1 -> p2."""
    x1, y1 = p1
    x2, y2 = p2
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    return np.degrees(angle_rad)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Points between eyes (head level indicator)
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        p1 = (int(left_eye.x * w), int(left_eye.y * h))
        p2 = (int(right_eye.x * w), int(right_eye.y * h))

        # Draw line between eyes
        cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # Calculate tilt angle
        tilt_angle = calculate_angle(p1, p2)

        cv2.putText(frame, f"Tilt: {tilt_angle:.1f} deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if abs(tilt_angle) > TILT_THRESHOLD_DEGREES:
            cv2.putText(frame, "HEAD TILT DETECTED", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("Head Tilt Detector", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()