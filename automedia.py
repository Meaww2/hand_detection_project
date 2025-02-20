import cv2 
import pyautogui
import time
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)

start_init = False 

prev = -1

# Threshold for detecting sliding (in pixels)
SLIDE_THRESHOLD = 150

# Store previous landmark positions
prev_landmarks = None

while True:
    end_time = time.time()
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    img = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    res_g = recognizer.recognize(image)

    if res_g.gestures:
        gesture_name = res_g.gestures[0][0].category_name  # Get top gesture

        if not(prev==gesture_name):
            if not(start_init):
                start_time = time.time()
                start_init = True
            
            else:
                if gesture_name == "Thumb_Up":
                    pyautogui.press("up")
                    cv2.putText(frm, "Thumbs Up Detected!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if gesture_name == "Thumb_Down":
                    pyautogui.press("down")
                    cv2.putText(frm, "Thumbs Down Detected!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if gesture_name == "Open_Palm":
                    pyautogui.press("space")
                    cv2.putText(frm, "Thumbs Down Detected!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if gesture_name == "Pointing_Up":
                    pyautogui.press("left")
                    cv2.putText(frm, "Thumbs Down Detected!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if gesture_name == "Victory":
                    pyautogui.press("right")
                    cv2.putText(frm, "Thumbs Down Detected!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                prev = gesture_name
                start_init = False
        if (end_time-start_time) > 0.5:
            prev=-1

    # Slide tracking
    if res.multi_hand_landmarks:

        hand_keyPoints = res.multi_hand_landmarks[0]

        for hand_landmarks in res.multi_hand_landmarks:
            # Get the wrist landmark (landmark 0)
            wrist_landmark = hand_landmarks.landmark[0]
            h, w, _ = frm.shape
            wrist_x, wrist_y = int(wrist_landmark.x * w), int(wrist_landmark.y * h)

            # Draw the wrist landmark
            cv2.circle(frm, (wrist_x, wrist_y), 10, (0, 255, 0), -1)
            # Check for sliding
            if prev_landmarks is not None:
                prev_x, prev_y = prev_landmarks
                displacement = np.sqrt((wrist_x - prev_x)**2 + (wrist_y - prev_y)**2)

                if displacement > SLIDE_THRESHOLD:
                    print("Hand sliding detected!")
                    cv2.putText(frm, "Sliding!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    pyautogui.hotkey("shift", "n")

            # Update previous landmarks
            prev_landmarks = (wrist_x, wrist_y)
        
        drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break