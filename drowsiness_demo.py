import cv2
import mediapipe as mp
import numpy as np

import config
from eye_state_predictor import predict_eye_state
from eye_utils import compute_ear, extract_eye_state
from overlay_utils import draw_eye_label, draw_center_text, draw_scaled_text, ui_scale_for_width
from drowsiness_logic import update_eye_counters, update_microsleep, update_perclos


def scale_frame(frame):
    if config.PROCESS_SCALE == 1.0:
        return frame
    return cv2.resize(frame, None, fx=config.PROCESS_SCALE, fy=config.PROCESS_SCALE, interpolation=cv2.INTER_AREA)


def init_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


def main():
    face_mesh = init_face_mesh()
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    closed_frames_threshold = int(fps * config.TIRED_THRESHOLD_SECONDS)
    frame_window = int(fps * config.FRAME_WINDOW_SECONDS)

    left_eye_closed_frames = 0
    right_eye_closed_frames = 0
    consecutive_closed_frames = 0
    eye_state_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = scale_frame(frame)
        h, w = frame.shape[:2]
        ui_scale = ui_scale_for_width(w)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        tired_basic = False
        tired_perclos = False
        tired_tilt = False

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            left_state, left_box, left_ok = extract_eye_state(
                frame,
                face_landmarks,
                config.LEFT_EYE_IDX,
                w,
                h,
                config.EYE_BOX_MIN,
                config.EYE_BOX_MARGIN,
                predict_eye_state,
            )
            right_state, right_box, right_ok = extract_eye_state(
                frame,
                face_landmarks,
                config.RIGHT_EYE_IDX,
                w,
                h,
                config.EYE_BOX_MIN,
                config.EYE_BOX_MARGIN,
                predict_eye_state,
            )

            left_ear = compute_ear(face_landmarks, config.LEFT_EYE_EAR_IDX, w, h, config.EYE_EAR_MIN_WIDTH)
            right_ear = compute_ear(face_landmarks, config.RIGHT_EYE_EAR_IDX, w, h, config.EYE_EAR_MIN_WIDTH)

            left_visible = left_ok and left_ear is not None
            right_visible = right_ok and right_ear is not None

            if left_visible and left_ear < config.EYE_EAR_CLOSED_THRESH:
                left_state = 0
            if right_visible and right_ear < config.EYE_EAR_CLOSED_THRESH:
                right_state = 0

            if left_box:
                if not left_visible:
                    draw_eye_label(frame, left_box, "Left: Away", (0, 255, 255), ui_scale)
                else:
                    color = (0, 255, 0) if left_state else (0, 0, 255)
                    draw_eye_label(
                        frame,
                        left_box,
                        f"Left: {'Open' if left_state else 'Closed'}",
                        color,
                        ui_scale,
                    )

            if right_box:
                if not right_visible:
                    draw_eye_label(frame, right_box, "Right: Away", (0, 255, 255), ui_scale)
                else:
                    color = (0, 255, 0) if right_state else (0, 0, 255)
                    draw_eye_label(
                        frame,
                        right_box,
                        f"Right: {'Open' if right_state else 'Closed'}",
                        color,
                        ui_scale,
                    )

            if left_visible and right_visible:
                left_eye_closed_frames, right_eye_closed_frames = update_eye_counters(
                    left_state, right_state, left_eye_closed_frames, right_eye_closed_frames
                )

                if (
                    left_eye_closed_frames >= closed_frames_threshold
                    and right_eye_closed_frames >= closed_frames_threshold
                ):
                    tired_basic = True
                    draw_scaled_text(
                        frame,
                        "TIRED (Closed Eyes)",
                        (30, int(50 * ui_scale)),
                        (0, 0, 255),
                        ui_scale,
                    )

                both_closed = int(left_state == 0 and right_state == 0)
                perclos = update_perclos(eye_state_history, both_closed, frame_window)
                if perclos > 0.4:
                    tired_perclos = True
                color = (0, 0, 255) if tired_perclos else (0, 255, 0)
                draw_scaled_text(
                    frame,
                    f"PERCLOS: {perclos:.2f}",
                    (30, int(90 * ui_scale)),
                    color,
                    ui_scale,
                    base_thickness=1,
                )

                consecutive_closed_frames = update_microsleep(consecutive_closed_frames, both_closed)
                if consecutive_closed_frames >= config.MICROSLEEP_THRESHOLD:
                    draw_center_text(frame, "MICROSLEEP DETECTED", (0, 0, 255), ui_scale)
            else:
                left_eye_closed_frames = 0
                right_eye_closed_frames = 0
                consecutive_closed_frames = 0

            p1 = (
                int(face_landmarks.landmark[33].x * w),
                int(face_landmarks.landmark[33].y * h),
            )
            p2 = (
                int(face_landmarks.landmark[263].x * w),
                int(face_landmarks.landmark[263].y * h),
            )
            tilt_angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            draw_scaled_text(
                frame,
                f"Tilt: {tilt_angle:.1f} deg",
                (30, int(170 * ui_scale)),
                (255, 255, 0),
                ui_scale,
                base_thickness=1,
            )
            if abs(tilt_angle) > config.TILT_THRESHOLD_DEGREES:
                tired_tilt = True
                draw_scaled_text(
                    frame,
                    "HEAD TILT DETECTED",
                    (30, int(210 * ui_scale)),
                    (0, 0, 255),
                    ui_scale,
                )

        cv2.imshow("Driver Drowsiness Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
