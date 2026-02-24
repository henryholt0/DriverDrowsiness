VIDEO_SOURCE = "data/test_clip_car_1.mp4"
PROCESS_SCALE = 1.0  # < 1.0 to speed up processing; 1.0 keeps full resolution

# Drowsiness thresholds
TIRED_THRESHOLD_SECONDS = 1.5
MICROSLEEP_THRESHOLD = 5  # consecutive frames
FRAME_WINDOW_SECONDS = 5

# Head pose
TILT_THRESHOLD_DEGREES = 15

# Eye geometry + visibility
EYE_EAR_CLOSED_THRESH = 0.19
EYE_EAR_MIN_WIDTH = 8
EYE_BOX_MIN = 12
EYE_BOX_MARGIN = 5

# Eye landmark indices
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380]
LEFT_EYE_EAR_IDX = (33, 133, 159, 145, 158, 153)
RIGHT_EYE_EAR_IDX = (362, 263, 386, 374, 385, 380)
