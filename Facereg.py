import cv2
import numpy as np
import os

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Directory containing the dataset of face images
dataset_path = 'dataset'

# Initialize lists for images and corresponding labels
images = []
labels = []

# Dictionary to map label indices to person names
names = {}
id = 0

# Loop through the dataset directory and its subdirectories
for subdir, dirs, files in os.walk(dataset_path):
    for name in dirs:
        # Map label indices to person names
        names[id] = name
        subject_path = os.path.join(subdir, name)
        # Loop through image files in each person's directory
        for filename in os.listdir(subject_path):
            img_path = os.path.join(subject_path, filename)
            # Read the image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(id)
        id += 1

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Define the desired width and height for face images
width = 130
height = 100

# Create LBPH face recognizer
model = cv2.face.LBPHFaceRecognizer_create()

# Train the face recognizer with the images and corresponding labels
model.train(images, labels)

# Video capture from webcam
webcam = cv2.VideoCapture(0)

# Loop to capture and process each frame
while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        face_roi = gray[y:y+h, x:x+w]

        # Resize the face ROI to the desired width and height
        face_resize = cv2.resize(face_roi, (width, height))

        # Predict the label and confidence for the resized face ROI
        label, confidence = model.predict(face_resize)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the predicted label and confidence level above the rectangle
        cv2.putText(frame, f'{names[label]} - {confidence:.2f}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Face Recognition', frame)

    # Check for the 'Esc' key to exit the loop
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
