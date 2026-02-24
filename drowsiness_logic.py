
def update_eye_counters(left_state, right_state, left_counter, right_counter):
    if left_state == 0:
        left_counter += 1
    else:
        left_counter = 0

    if right_state == 0:
        right_counter += 1
    else:
        right_counter = 0

    return left_counter, right_counter


def update_perclos(eye_state_history, both_closed, frame_window):
    eye_state_history.append(both_closed)
    if len(eye_state_history) > frame_window:
        eye_state_history.pop(0)
    return sum(eye_state_history) / len(eye_state_history)


def update_microsleep(consecutive_closed_frames, both_closed):
    if both_closed:
        consecutive_closed_frames += 1
    else:
        consecutive_closed_frames = 0
    return consecutive_closed_frames
