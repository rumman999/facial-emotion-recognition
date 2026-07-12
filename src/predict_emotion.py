import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

def predict_emotion(image_path):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/emotion_model.keras")
    model = load_model(model_path)
    labels = ['angry', 'happy', 'neutral', 'sad', 'surprise']
    
    # 1. Load image and Detect Face
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected"

    # 2. Crop, Resize, and Normalize
    x, y, w, h = faces[0]
    face_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
    input_img = (np.array(face_roi, 'float32') / 255.0).reshape(1, 48, 48, 1)

    # 3. Predict
    prediction = model.predict(input_img)
    return labels[np.argmax(prediction)]

# Usage
print(predict_emotion("D:\\code\\python\\facial-emotion-recognition\\data\\samples\\sad.jpg"))