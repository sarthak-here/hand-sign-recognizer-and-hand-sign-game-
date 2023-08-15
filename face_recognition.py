import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow
import keras
from videosource import WebcamSource
from tensorflow import keras
from keras.models import load_model

DATASET_DIR = 'path/to/C:/Users/sarthak/gui/trained pics'
model_path = "path/to/your/model.h5"

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

# Define paths for face recognition model and dataset


# Load face recognition model
model = load_model(model_path)

# Load dataset
dataset = {}
for person_dir in os.listdir(DATASET_DIR):
    person_id = len(dataset)
    for img_file in os.listdir(os.path.join(DATASET_DIR, person_dir)):
        img_path = os.path.join(DATASET_DIR, person_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dataset[person_id] = img

# Create labels for dataset
labels = np.array(list(dataset.keys()))

# Create FaceDetection and WebcamSource instances
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
source = WebcamSource()

# Define threshold for face recognition confidence
THRESHOLD = 0.6

# Main loop for real-time face recognition
while True:
    # Capture frame from webcam
    frame, frame_rgb = next(source)

    # Detect faces in frame using FaceDetection
    results = face_detection.process(frame_rgb)

    # Loop over detected faces
    if results.detections:
        for detection in results.detections:
            # Extract face image from frame
            bbox = detection.location_data.relative_bounding_box
            ymin, xmin, height, width = bbox.ymin, bbox.xmin, bbox.height, bbox.width
            ymin, xmin, ymax, xmax = int(ymin * frame.shape[0]), int(xmin * frame.shape[1]), int(
                (ymin + height) * frame.shape[0]), int((xmin + width) * frame.shape[1]
            )
            face_img = frame_rgb[ymin:ymax, xmin:xmax]

            # Preprocess face image for face recognition model
            face_img = cv2.resize(face_img, (160, 160))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            # Predict identity of face using face recognition model
            embedding = model.predict(face_img)
            # Compute distance between face embedding and dataset embeddings
            distances = np.linalg.norm(embedding - dataset, axis=1)

            # Identify person with closest embedding
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            if min_distance < THRESHOLD:
                # Draw bounding box and label with person's name
                person_id = labels[min_distance_idx]
                label = f"Person {person_id}"
                mp_drawing.draw_detection(frame, detection)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    source.show(frame)