import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load and play a song
pygame.mixer.music.load("mp3.mp3")
pygame.mixer.music.play(-1)  # -1 means the song will loop indefinitely
pygame.mixer.music.set_volume(0.5)  # Set initial volume to 50%

# Initialize camera
cap = cv2.VideoCapture(0) #class

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

isMuted = False

def calculate_distance(x1, y1, x2, y2):
    return hypot(x2 - x1, y2 - y1)

# Initialize a variable to keep track of frames
frame_count = 0
FRAME_SKIP = 2  # Process every 2nd frame

while True:
    success, img = cap.read()
    if not success:
        break
    
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # Skip this frame to reduce processing load

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList:
        # Thumb and index finger tips
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Calculate the length between thumb and index finger
        length = calculate_distance(x1, y1, x2, y2)

        # Draw circles and line
        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        if not isMuted:
            # Hand range 15 - 220
            # Volume range 0.0 - 1.0 (Pygame mixer volume range)
            vol = np.interp(length, [15, 220], [0.0, 1.0])
            pygame.mixer.music.set_volume(vol)
            cv2.putText(img, f'Volume: {int(vol * 100)}%', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print(f"Volume: {vol}, Length: {length}")

        # Check if the hand is open (all fingers spread apart)
        thumb_index_dist = calculate_distance(lmList[4][1], lmList[4][2], lmList[8][1], lmList[8][2])
        index_middle_dist = calculate_distance(lmList[8][1], lmList[8][2], lmList[12][1], lmList[12][2])
        middle_ring_dist = calculate_distance(lmList[12][1], lmList[12][2], lmList[16][1], lmList[16][2])
        ring_pinky_dist = calculate_distance(lmList[16][1], lmList[16][2], lmList[20][1], lmList[20][2])

        # Gesture: Open hand to mute, closed fist to unmute
        if (thumb_index_dist > 50 and index_middle_dist > 50 and middle_ring_dist > 50 and ring_pinky_dist > 50):
            if not isMuted:
                pygame.mixer.music.set_volume(0)  # Mute
                isMuted = True
                cv2.putText(img, 'Muted', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Muted")
        else:
            if isMuted:
                pygame.mixer.music.set_volume(0.5)  # Unmute to default 50%
                isMuted = False
                cv2.putText(img, 'Unmuted', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("Unmuted")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
