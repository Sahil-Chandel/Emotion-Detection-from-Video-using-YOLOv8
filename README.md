<h1>Emotion-Detection-from-Video-using-YOLOv8</h1>

# Full Workflow Code for Emotion Detection and Recognition

## Project Structure


emotion-detection/

- data/                  # Store datasets
  - wider_face/          # WIDER FACE dataset
  - fer2013/             # FER-2013 dataset
  - affectnet/           # AffectNet dataset

- models/                # Trained and fine-tuned models
  - yolov5/              # YOLO face detection model
  - emotion_classifier/  # Emotion recognition model

- src/                   # Source code
  - preprocess/          # Preprocessing code
    - face_detection.py
    - video_preprocessor.py
  - train/               # Training scripts
    - train_yolo.py
    - train_emotion.py
  - pipeline/            # Model pipeline
    - emotion_pipeline.py
  - api/                 # API scripts
    - app.py
  - ui/                  # Real-time web interface
    - templates/
      - index.html
    

- requirements.txt       # Python dependencies
- config.yaml            # Configuration settings
- README.md              # Project documentation
```
