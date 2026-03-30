#Import necessary libraries
import cv2
import mediapipe as mp
import time
import math
from PIL import ImageGrab
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

#Load hand landmarker model
model_path = "/Users/macbookie/Documents/Work/3.Personal/Upskilling/Python/Git Projects/HandGesture/hand_landmarker.task"

def is_pinch(hand_landmarks, threshold=0.05):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]

    dx = thumb_tip.x - index_tip.x
    dy = thumb_tip.y - index_tip.y
    dz = thumb_tip.z - index_tip.z  # Optional: depth if available

    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    return distance < threshold

def take_screenshot(frame):
    timestamp = int(time.time())  # Unique timestamp
    filename = f"/Users/macbookie/Documents/Work/3.Personal/Upskilling/Python/Git Projects/HandGesture/assets/screenshot_{timestamp}.png"
    screenshot = ImageGrab.grab()
    screenshot.save(filename)
    print(f"Screenshot saved as {filename}")

#Initisalise the HandLandmarker, using 2 hands
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=2
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success or frame is None:
        print("Failed to access camera")
        break

    frame = cv2.flip(frame, 1)

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Perform detection
    result = hand_landmarker.detect(mp_image)

    # Finger colors (BGR)
    finger_colors = {
        'thumb': (255, 0, 0),   # Blue
        'index': (0, 0, 255),   # Red
        'middle': (0, 255, 255),  # Yellow 
        'ring': (0, 255, 0),  # Green
        'pinky': (255, 0, 255)  # Magenta
    }

    # Map landmark index to finger
    landmark_to_finger = {
        1: 'thumb', 2: 'thumb', 3: 'thumb', 4: 'thumb',
        5: 'index', 6: 'index', 7: 'index', 8: 'index',
        9: 'middle', 10: 'middle', 11: 'middle', 12: 'middle',
        13: 'ring', 14: 'ring', 15: 'ring', 16: 'ring',
        17: 'pinky', 18: 'pinky', 19: 'pinky', 20: 'pinky'
    }

    # Draw landmarks manually
    for hand_landmarks in result.hand_landmarks:
        for idx, landmark in enumerate(hand_landmarks):
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)

            if idx == 0:  # wrist
                color = (200, 200, 200)  # Gray for wrist
            else:
                finger = landmark_to_finger[idx]
                color = finger_colors[finger]

            cv2.circle(frame, (cx, cy), 5, color, -1)

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) == 27:
        break


    last_screenshot_time = 0
    screenshot_cooldown = 2  # seconds

    for hand_landmarks in result.hand_landmarks:
        if is_pinch(hand_landmarks):
            gesture = "Pinch"
            current_time = time.time()
            if current_time - last_screenshot_time > screenshot_cooldown:
                take_screenshot(frame)
                last_screenshot_time = current_time


cap.release()
cv2.destroyAllWindows()