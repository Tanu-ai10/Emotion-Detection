import cv2
from keras.models import model_from_json
import numpy as np
import sys

# Ensure console uses UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load model architecture
json_file = open(r"C:\Users\lenovo\Desktop\coding\Facedetection\emotiondetector.json", encoding="utf-8", errors="ignore")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load model weights
model.load_weights(r"C:\Users\lenovo\Desktop\coding\Facedetection\emotiondetector.h5")

# Load OpenCV face detector
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Open webcam
webcam = cv2.VideoCapture(0)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Ensure correct encoding for OpenCV text rendering
            cv2.putText(im, prediction_label.encode("utf-8").decode("utf-8"), (p-10, q-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0))

        cv2.imshow("Output", im)

        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error as e:
        print(f"OpenCV error: {e}")

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()
