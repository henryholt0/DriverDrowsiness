import numpy as np
import cv2
import time
last_display_time = 0

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in eyes[:2]:
        eye_roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        current_time = time.time()
        if current_time - last_display_time > 10:
            cv2.imshow(f"Eye ROI ({x},{y})", eye_roi)
            last_display_time = current_time

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    cv2.imshow("test", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import numpy as np
# import cv2
# import time
# last_display_time = 0

# eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in eyes[:2]:
#         eye_roi = frame[y:y+h, x:x+w]
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         current_time = time.time()
#         if current_time - last_display_time > 10:
#             cv2.imshow(f"Eye ROI ({x},{y})", eye_roi)
#             last_display_time = current_time

#     blurred = cv2.GaussianBlur(frame, (5, 5), 0)
#     edges = cv2.Canny(blurred, 30, 100)

#     cv2.imshow("test", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
