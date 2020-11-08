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
    # Load and return the model
    return load_model(path)

# The function to detect the gender
def detectGender(image, model):
    if image is None:
        print("Could not read input image")
        exit()
    # Detect faces in the image
    face, confidence = cv.detect_face(image)
    classes = ['Male', 'Female']
    # Take the very first face
    if len(face) > 0:
        (startX, startY) = face[0][0], face[0][1]
        (endX, endY) = face[0][2], face[0][3]
    else:
        return "Couldn't detect a face"
    # Crop the face
    croppedFace = np.copy(image[startY:endY, startX:endX])
    # Size and normalize te face
    croppedFace = cv2.resize(croppedFace, (96, 96))
    croppedFace = croppedFace.astype("float") / 255.0
    croppedFace = img_to_array(croppedFace)
    croppedFace = np.expand_dims(croppedFace, axis=0)
    # Predict the gender
    conf = model.predict(croppedFace)[0]
    # Get the label which has the maximum accuracy
    idx = np.argmax(conf)
    label = classes[idx]
    return label + " | Confidence: " + str(int(max(conf[0], conf[1]) * 100)) + "%"


