import cv2
import numpy as np


def compute_ear(face_landmarks, indices, w, h, min_width):
    pts = [
        np.array([face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h])
        for i in indices
    ]
    p1, p4, p2, p6, p3, p5 = pts
    width = np.linalg.norm(p1 - p4)
    if width < min_width:
        return None
    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * width)
    return float(ear)


def extract_eye_state(frame, face_landmarks, indices, w, h, box_min, box_margin, predict_fn):
    pts = [
        (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
        for i in indices
    ]
    x, y, bw, bh = cv2.boundingRect(np.array(pts))
    x1 = max(x - box_margin, 0)
    y1 = max(y - box_margin, 0)
    x2 = min(x + bw + box_margin, w)
    y2 = min(y + bh + box_margin, h)
    if (x2 - x1) < box_min or (y2 - y1) < box_min:
        return None, (x1, y1, x2, y2), False

    eye_img = frame[y1:y2, x1:x2]
    if eye_img.size == 0:
        return None, (x1, y1, x2, y2), False

    state = predict_fn(eye_img)
    return state, (x1, y1, x2, y2), True
