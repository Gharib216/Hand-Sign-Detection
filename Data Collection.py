import cv2 as cv
import HandTrackingModule as htm
import numpy as np
import time

cap = cv.VideoCapture(0)
detector = htm.HandDetector(max_num_hands=1, min_tracking_confidence=0.7)
previous_time = 0

# Constants
OFFSET = 15
BACKGROUND_SIZE = (300, 300)
folder = 'Data/C'
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

        if handsInfo:
            background = np.full((BACKGROUND_SIZE[0], BACKGROUND_SIZE[1], 3), 255, dtype='uint8')
            hand = handsInfo[0]
            x1, y1, x2, y2 = hand["bbox"]
            imgCrop = frame[y1 - OFFSET : y2 + OFFSET, x1 - OFFSET : x2 + OFFSET]

            height, width = imgCrop.shape[:2]
            
            if height == 0 or width == 0:
                continue
             
            aspect_ratio = width / height

            if aspect_ratio > 1:
                new_width = BACKGROUND_SIZE[1]
                new_height = int(BACKGROUND_SIZE[1] / aspect_ratio)
            else:
                new_height = BACKGROUND_SIZE[0]
                new_width = int(BACKGROUND_SIZE[0] * aspect_ratio)


            new_width = min(new_width, BACKGROUND_SIZE[1])
            new_height = min(new_height, BACKGROUND_SIZE[0])

            imgResize = cv.resize(imgCrop, (new_width, new_height))

            x_offset = (BACKGROUND_SIZE[1] - new_width) // 2
            y_offset = (BACKGROUND_SIZE[0] - new_height) // 2

            background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = imgResize
   
            cv.imshow('Hand Img', background)

            key = cv.waitKey(1)

            if key == ord('s'):
                cv.imwrite(f'{folder}/Image_{counter}.jpg', background)
                counter += 1


        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv.putText(frame, f'FPS: {int(fps)}', (10, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv.imshow('Live Cam', frame)

        if cv.waitKey(1) == 27:
            break

cap.release()
cv.destroyAllWindows()