import cv2
import mediapipe as mp
import math
import numpy as np
import streamlit as st
import platform

# Initialize MediaPipe Hands and Audio Utilities for volume control
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Volume control setup (only for Windows)
volume, minVol, maxVol = None, None, None
if platform.system() == "Windows":
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol, maxVol = volRange[0], volRange[1]

# Streamlit app configuration
st.title("Hand Gesture Volume Control")
st.text("Show your hand to control the system volume (Windows only).")

# Streamlit camera input widget for webcam
camera_input = st.camera_input("Take a picture")

# Check if a video frame is captured
if camera_input:
    # Convert the captured image to an OpenCV format (numpy array)
    img = camera_input.getvalue()
    image = np.array(bytearray(img), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Process the image for hand tracking
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ).process(image_rgb)
    
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    lmList = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Get landmark list for the first detected hand
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image_bgr.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

    # Calculate distance between thumb and index finger
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger
        cv2.circle(image_bgr, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
        cv2.circle(image_bgr, (x2, y2), 15, (255, 255, 255), cv2.FILLED)
        cv2.line(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        length = math.hypot(x2 - x1, y2 - y1)

        # Change line color if distance is below threshold
        if length < 50:
            cv2.line(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Volume control based on the distance (only on Windows)
        if platform.system() == "Windows" and volume:
            vol = np.interp(length, [50, 220], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)
            volBar = np.interp(length, [50, 220], [400, 150])
            volPer = np.interp(length, [50, 220], [0, 100])

            # Volume bar on the side of the screen
            cv2.rectangle(image_bgr, (50, 150), (85, 400), (0, 0, 0), 3)
            cv2.rectangle(image_bgr, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
            cv2.putText(image_bgr, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

    # Display the image with detected hand and volume controls
    st.image(image_bgr, channels="BGR")
else:
    st.write("Please enable the camera to use the app.")
