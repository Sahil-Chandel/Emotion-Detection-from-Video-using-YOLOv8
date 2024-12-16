# Purpose : Integrate face detection and emotion classification into a single pipeline.


import cv2
from src.preprocess.face_detection import detect_faces
from tensorflow.keras.models import load_model
import numpy as np

# Load emotion classifier model
emotion_model = load_model("./models/emotion_classifier/emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']

def classify_emotions(face_image):
    resized_face = cv2.resize(face_image, (48, 48))
    resized_face = resized_face.reshape(1, 48, 48, 1) / 255.0
    prediction = emotion_model.predict(resized_face)
    return emotion_labels[np.argmax(prediction)]

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame)
        for face in faces:
            x1, y1, x2, y2 = map(int, face)
            face_image = frame[y1:y2, x1:x2]
            if face_image.size > 0:
                emotion = classify_emotions(face_image)
                cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        out.write(frame)

    cap.release()
    out.release()