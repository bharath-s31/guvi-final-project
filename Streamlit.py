import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Flatten(),  
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

model = EmotionCNN(num_classes=7)
model.load_state_dict(torch.load('best_model_emotion.pth', map_location=torch.device('cpu')))
model.eval()

def extract_landmarks(img):
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
    faces = detector(gray)  
    if len(faces) == 0:
        return Image.fromarray(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(68):  
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(gray, (x, y), 2, (255, 255, 255), -1) 
    return Image.fromarray(gray)  

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.Lambda(lambda img: extract_landmarks(img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

st.title("Facial Emotion Recognition")
st.write("Upload an image to classify the facial emotion.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        st.write(f"Predicted Emotion: **{classes[predicted.item()]}**")
