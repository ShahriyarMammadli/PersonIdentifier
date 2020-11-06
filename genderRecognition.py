# Shahriyar Mammadli
# Import required libraries
from keras.models import load_model
from keras.utils import get_file
import os
import cvlib as cv
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

# Function to load pre-trained model
def loadGenderModel():
    # Source of the model
    source = "https://s3.ap-south-1.amazonaws.com/arunponnusamy/pre-trained-weights/gender_detection.model"
    path = get_file("gender_detection.model", source,
                          cache_subdir="pre-trained", cache_dir=os.getcwd())
    # load model
    return load_model(path)

def detectImage(image, model):
    if image is None:
        print("Could not read input image")
        exit()
    # Detect faces in the image
    face, confidence = cv.detect_face(image)
    classes = ['Male', 'Female']
    # loop through detected faces
    for idx, f in enumerate(face):
        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(image[startY:endY, startX:endX])

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]
        # print(conf)
        # print(classes)

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        return label + " | Confidence: " + str(int(max(conf[0], conf[1]) * 100)) + "%"


