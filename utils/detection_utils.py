import cv2
import numpy as np
import os

haar_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = haar_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return [[x, y, x + w, y + h] for (x, y, w, h) in boxes]

def detect_ssd(image, conf_threshold=0.5):
    model_path = "models"
    proto = os.path.join(model_path, "deploy.prototxt")
    weights = os.path.join(model_path, "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe(proto, weights)

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            boxes.append(box.astype(int).tolist())
    return boxes