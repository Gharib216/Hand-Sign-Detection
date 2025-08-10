import cv2 as cv
import HandTrackingModule as htm
import numpy as np
import time
import os

# ===== Configuration =====
OFFSET = 15
BACKGROUND_SIZE = (300, 300)
DATA_FOLDER = 'Data'

MAX_IMAGES = 2000           # Max images for ASL letters/digits
MAX_NEGATIVE_IMAGES = 3000  # Max images for "NEGATIVE" class

CATEGORIES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'NEGATIVE'
]

counter_dict = {}

# Count existing images per category (only once at start)
for cat in CATEGORIES:
    path = os.path.join(DATA_FOLDER, cat)
    os.makedirs(path, exist_ok=True)
    existing_files = [file for file in os.listdir(path) if file.lower().endswith(('.jpg', '.png'))]
    counter_dict[cat] = len(existing_files) + 1

KEY_CODES = {
    "UP": 2490368,
    "DOWN": 2621440,
    "LEFT": 2424832,
    "RIGHT": 2555904,
    "ESC": 27
}

def caputureHandImg(img, bbox, offset, bg_size):
    """Crop hand from image and fit into background size."""
    x1, y1, x2, y2 = bbox
    cropped = img[y1 - offset:y2 + offset, x1 - offset:x2 + offset]
    h, w = cropped.shape[:2]

    if h == 0 or w == 0:
        return None

    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = bg_size[1]
        new_h = int(bg_size[1] / aspect_ratio)
    else:
        new_h = bg_size[0]
        new_w = int(bg_size[0] * aspect_ratio)

    new_w = min(new_w, bg_size[1])
    new_h = min(new_h, bg_size[0])

    resized = cv.resize(cropped, (new_w, new_h))
    background = np.full((bg_size[0], bg_size[1], 3), 255, dtype='uint8')

    x_offset = (bg_size[1] - new_w) // 2
    y_offset = (bg_size[0] - new_h) // 2
    background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return background

# ===== Init =====
cap = cv.VideoCapture(0)
detector = htm.HandDetector(max_num_hands=1, min_tracking_confidence=0.7)
previous_time = 0
pointer = 10  # start at 'A'
folder = f"Data/{CATEGORIES[pointer]}"
counter = 1


if not cap.isOpened():
    print("Error: Could not open the camera.")
else:
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv.flip(frame, 1)
        handsInfo, frame = detector.findHands(frame)

        key = cv.waitKeyEx(1)

        if handsInfo:
            hand = handsInfo[0]
            hand_img = caputureHandImg(frame, hand['bbox'], OFFSET, BACKGROUND_SIZE)
            
            if hand_img is not None:
                cv.imshow('Hand Img', hand_img)

            # key = cv.waitKey(1)

            if key == ord('s'):
                count = counter_dict[CATEGORIES[pointer]]
                
                if CATEGORIES[pointer] == 'NEGATIVE':
                    max_limit = MAX_NEGATIVE_IMAGES
                else:
                    max_limit = MAX_IMAGES
                
                if count < max_limit:
                    cv.imwrite(f'{folder}/Image_{count}.jpg', hand_img)
                    counter_dict[CATEGORIES[pointer]] += 1
                    print(f"[SAVED] {folder}/Image_{count}.jpg")
                else:
                    print(f"[INFO] Max images reached for {CATEGORIES[pointer]} ({max_limit}).")

        # key = cv.waitKeyEx(1)

        if key == KEY_CODES['UP']:  # UP
            pointer = (pointer - 10) % len(CATEGORIES)
        elif key == KEY_CODES['DOWN']:  # DOWN
            pointer = (pointer + 10) % len(CATEGORIES)
        elif key == KEY_CODES['LEFT']:  # LEFT
            pointer = (pointer - 1) % len(CATEGORIES)
        elif key == KEY_CODES['RIGHT']:  # RIGHT
            pointer = (pointer + 1) % len(CATEGORIES)
        
        folder = f"Data/{CATEGORIES[pointer]}"

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv.putText(frame, f'FPS: {int(fps)}', (10, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.putText(frame, f'Folder: {folder}', (10, 90), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        cv.imshow('Live Cam', frame)

        if key == KEY_CODES['ESC']:
            break

cap.release()
cv.destroyAllWindows()
