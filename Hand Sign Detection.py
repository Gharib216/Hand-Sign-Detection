import cv2 as cv
import HandTrackingModule as htm
import numpy as np
import time
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import joblib


class SignLanguageClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128 * 16 * 16), 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        for conv in [self.conv1, self.conv2, self.conv3]:
            x = self.relu(self.pooling(conv(x)))
        x = self.flatten(x)
        x = self.linear(x)
        return self.output(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

encoder = joblib.load('label_encoder.pkl')
idx_to_label = {i: label for i, label in enumerate(encoder.classes_)}
num_classes = len(idx_to_label)

model = SignLanguageClassifier(num_classes).to(device)
model.load_state_dict(torch.load('sign_language_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])


def predict(frame, threshold=0.9):
    image = Image.fromarray(frame)
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        max_prob, pred_class = torch.max(probabilities, dim=1)
        if max_prob.item() < threshold:
            return "Unknown"
        return idx_to_label[pred_class.item()]


cap = cv.VideoCapture(0)
detector = htm.HandDetector(max_num_hands=1, min_tracking_confidence=0.7)
previous_time = 0

OFFSET = 15
BACKGROUND_SIZE = (300, 300)

if not cap.isOpened():
    print("Error: Could not open the camera.")
else:
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv.flip(frame, 1)
        handsInfo, frame = detector.findHands(frame, draw=False)

        if handsInfo:
            background = np.full((BACKGROUND_SIZE[0], BACKGROUND_SIZE[1], 3), 255, dtype='uint8')
            hand = handsInfo[0]
            x1, y1, x2, y2 = hand["bbox"]
            imgCrop = frame[y1 - OFFSET:y2 + OFFSET, x1 - OFFSET:x2 + OFFSET]

            h, w = imgCrop.shape[:2]
            if h == 0 or w == 0:
                continue

            aspect_ratio = w / h
            if aspect_ratio > 1:
                new_w = BACKGROUND_SIZE[1]
                new_h = int(BACKGROUND_SIZE[1] / aspect_ratio)
            else:
                new_h = BACKGROUND_SIZE[0]
                new_w = int(BACKGROUND_SIZE[0] * aspect_ratio)

            new_w = min(new_w, BACKGROUND_SIZE[1])
            new_h = min(new_h, BACKGROUND_SIZE[0])

            imgResize = cv.resize(imgCrop, (new_w, new_h))
            x_offset = (BACKGROUND_SIZE[1] - new_w) // 2
            y_offset = (BACKGROUND_SIZE[0] - new_h) // 2

            background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = imgResize

            label = predict(background)
            cv.putText(frame, f'{label}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv.putText(frame, f'FPS: {int(fps)}', (10, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv.imshow('Sign Language Live', frame)
        if cv.waitKey(1) == 27:
            break

cap.release()
cv.destroyAllWindows()
