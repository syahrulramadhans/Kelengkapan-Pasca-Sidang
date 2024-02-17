import cv2
import os

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Initialize face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')
# Prompt user for name
# name = "syahrul"
name = str(input("Masukkan nama: "))

# Create directory for faces if it doesn't exist
if not os.path.exists('faces'):
    os.makedirs('faces')

# Create directory for current user's faces if it doesn't exist
if not os.path.exists(f'faces/{name}'):
    os.makedirs(f'faces/{name}')

# Set counter for number of faces captured
face_num = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # vertical flip

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Capture Faces', frame)

    # Press 's' to save face
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Save face image
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (64, 48))
        cv2.imwrite(f'faces/{name}/{name}_{face_num}.jpg', face_img)

        # Increment face counter
        face_num += 1

    # Press 'q' to quit
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
