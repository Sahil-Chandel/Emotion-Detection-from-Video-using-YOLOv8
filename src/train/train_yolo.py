# Purpose : Fine-tune YOLO for face detection using the WIDER FACE dataset.


import os
from ultralytics import YOLO

# Paths
data_path = "notebook\data\WIDER_FACE"
model_path = "./models/yolov8/face_detection.pt"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use a lightweight model for fine-tuning

# Train model
model.train(data=f"{data_path}/data.yaml", epochs=50, imgsz=640)

# Save model
model.export(format="torchscript", path=model_path)