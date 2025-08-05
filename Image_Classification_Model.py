import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import joblib


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label = torch.tensor(encoder.transform(dataframe['Label'].values), dtype=torch.long)

    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, index):
        image_path = self.dataframe.iloc[index, 0]
        label = self.label[index]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class SignLanguageClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128*16*16), 128)

        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        for conv in [self.conv1, self.conv2, self.conv3]:
            x = self.relu(self.pooling(conv(x)))
            
        x = self.flatten(x)
        x = self.linear(x)

        return self.output(x)
    

# Constants
LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 16 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Devices Available: {device}')

image_path = []
label = []

for folder in os.listdir('Data'):
    for image in os.listdir(f'Data/{folder}'):
        image_path.append(f'Data/{folder}/{image}')
        label.append(folder)

images_df = pd.DataFrame(zip(image_path, label), columns=['Image Path', 'Label'])

train_df, test_df = train_test_split(images_df, train_size=0.7, random_state=42)

encoder = LabelEncoder()
encoder.fit(images_df['Label'])

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

train_dataset = CustomImageDataset(dataframe=train_df, transform=transform)
test_dataset = CustomImageDataset(dataframe=test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(images_df['Label'].unique())
model = SignLanguageClassifier(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f} | Accuracy: {100 * total_correct / total_samples:.2f}%")

model.eval()
correct = 0
total = 0
loss_total = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_total += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

print(f"Test Loss: {loss_total / len(test_loader):.4f} | Test Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), 'sign_language_model.pth')

joblib.dump(encoder, 'label_encoder.pkl')