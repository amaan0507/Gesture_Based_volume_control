import cv2
import mediapipe as mp
import math
import numpy as np
import streamlit as st
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


import platform

if platform.system() == "Windows":
    try:
        from comtypes import CLSCTX_ALL
        from ctypes import cast, POINTER
        # Add any other Windows-specific imports here
    except ImportError:
        print("comtypes is not available, skipping Windows-specific features.")
else:
    print("Non-Windows environment detected, skipping Windows-specific imports.")


# Initialize MediaPipe Hands and Audio Utilities for volume control
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]
volBar, volPer = 400, 0

# Streamlit app configuration
st.title("Hand Gesture Volume Control")
st.text("Show your hand to control the system volume.")

# Initialize Webcam
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# Run MediaPipe Hands for hand gesture recognition
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Streamlit container for video
    video_container = st.empty()

    while True:
        success, image = cam.read()
        if not success:
            break

        # Process the image for hand tracking
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        lmList = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Get landmark list for the first detected hand
                myHand = results.multi_hand_landmarks[0]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

        # Calculate distance between thumb and index finger
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
            x2, y2 = lmList[8][1], lmList[8][2]  # Index finger
            cv2.circle(image, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (255, 255, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            length = math.hypot(x2 - x1, y2 - y1)

            # Change line color if distance is below threshold
            if length < 50:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Volume control based on the distance
            vol = np.interp(length, [50, 220], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)
            volBar = np.interp(length, [50, 220], [400, 150])
            volPer = np.interp(length, [50, 220], [0, 100])

            # Volume bar on the side of the screen
            cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
            cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

        # Display the image with detected hand and volume controls
        video_container.image(image, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
