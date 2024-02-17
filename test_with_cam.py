import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load face data
faces = []
labels = []
for label, name in enumerate(os.listdir('faces')):
    for image_name in os.listdir(f'faces/{name}'):
        image = cv2.imread(f'faces/{name}/{image_name}')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to match the input size of the camera
        resized = cv2.resize(gray, (64, 48))

        faces.append(resized.flatten())
        labels.append(label)

# Create KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')

# Train the classifier
X_train = np.array(faces)
y_train = np.array(labels)
knn.fit(X_train, y_train)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

minWidth = 0.1*cap.get(3)
minHeight = 0.1*cap.get(4)
 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # vertical flip

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the faces and predict labels
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Resize the face region to match the input size of the model
        face_roi = gray[y:y+h, x:x+w]
        resized_face_roi = cv2.resize(face_roi, (64, 48))

        # Predict label for the face region]

        # Predict label and confidence score for the face region
        # label, confidence = knn.predict_proba(
        #     resized_face_roi.flatten().reshape((1, -1)))[0]

        # confidence_percent = round(confidence * 100, 2)

        # Display the predicted name and confidence score on the frame
        # text = f"{name}: {confidence_percent}% confidence"
        # cv2.putText(frame, text, (x+5, y-25),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        label = knn.predict(resized_face_roi.flatten().reshape((1, -1)))[0]

        name = os.listdir('faces')[label]

        confidence = knn.predict_proba(
            resized_face_roi.flatten().reshape((1, -1)))[0][label]

        confidence_percent = round(confidence * 100, 2)
        # Display predicted name and confidence on the frame
        text = f"{name}: {confidence_percent}% confidence"
        cv2.putText(frame, text, (x+5, y-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
