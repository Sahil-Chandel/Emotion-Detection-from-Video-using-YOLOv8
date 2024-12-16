# Purpose : Process video frames to detect faces and draw bounding boxes.


import cv2
from src.preprocess.face_detection import detect_faces

def preprocess_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame)
        for face in faces:
            x1, y1, x2, y2 = map(int, face)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        out.write(frame)

    cap.release()
    out.release()