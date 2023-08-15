import cv2
import numpy as np
import mediapipe as mp
import os
from keras import layers
from keras.models import load_model
from keras import models


# Load the face detection model from Mediapipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
DATASET_DIR = 'path/to/C:/Users/sarthak/gui/trained pics'
MODEL_PATH = 'path/to/face_recognition_model.h5'

# Create a dictionary to store the faces of each person in the dataset
dataset = {}

# Define a function to extract and preprocess faces from an image
def extract_faces(img):
    # Detect faces in the image
    results = face_detection.process(img)
    if results.detections:
        # Loop over the detected faces
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ymin, xmin, height, width = bbox.ymin, bbox.xmin, bbox.height, bbox.width
            ymin, xmin, ymax, xmax = int(ymin * img.shape[0]), int(xmin * img.shape[1]), int(
                (ymin + height) * img.shape[0]), int((xmin + width) * img.shape[1]
            )
            face_img = img[ymin:ymax, xmin:xmax]
            # Preprocess the face image
            face_img = cv2.resize(face_img, (160, 160))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            return face_img
    return None

# Loop over the images in the dataset directory
for person_dir in os.listdir(DATASET_DIR):
    person_id = len(dataset)
    for img_file in os.listdir(os.path.join(DATASET_DIR, person_dir)):
        img_path = os.path.join(DATASET_DIR, person_dir, SARTHAK.jpg)
        img = cv2.imread(img_path)
        # Extract and preprocess the face from the image
        face_img = extract_faces(img)
        if face_img is not None:
            # Add the face to the dataset
            if person_id in dataset:
                dataset[person_id] = np.concatenate([dataset[person_id], face_img], axis=0)
            else:
                dataset[person_id] = face_img

# Create arrays for the faces and labels in the dataset
faces = []
labels = []
for label, face_images in dataset.items():
    for face in face_images:
        faces.append(face)
        labels.append(label)

# Convert the arrays to numpy arrays
faces = np.array(faces)
labels = np.array(labels)

# Shuffle the data
permutation = np.random.permutation(len(faces))
faces = faces[permutation]
labels = labels[permutation]
def create_model():
    model = load_model(MODEL_PATH)          
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='linear'))

    model.compile(optimizer='adam', loss='mse')

    return model

# Create the face recognition model
model = create_model()

# Train the model on the dataset
model.fit(faces, labels, epochs=10, batch_size=16)

# Save the model to a file
model.save(MODEL_PATH)
