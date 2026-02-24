import cv2


def ui_scale_for_width(width):
    return max(0.6, width / 1280.0)


def draw_eye_label(frame, box, label, color, ui_scale):
    if not box:
        return
    x1, y1, _, _ = box
    cv2.putText(
        frame,
        label,
        (x1, y1 - int(10 * ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5 * ui_scale,
        color,
        max(1, int(1 * ui_scale)),
    )


def draw_scaled_text(frame, text, pos, color, ui_scale, base_scale=0.7, base_thickness=2):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        base_scale * ui_scale,
        color,
        max(1, int(base_thickness * ui_scale)),
    )


def draw_center_text(frame, text, color, ui_scale):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9 * ui_scale
    thickness = max(2, int(2 * ui_scale))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (w - tw) // 2)
    y = max(th + 10, (h + th) // 2)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)
