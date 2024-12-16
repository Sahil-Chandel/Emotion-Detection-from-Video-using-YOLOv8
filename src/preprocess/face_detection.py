# Purpose : Perform face detection using the YOLOv8 model.


from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("./models/yolov8/face_detection.pt")

def detect_faces(frame):
    results = model.predict(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes
    return boxes