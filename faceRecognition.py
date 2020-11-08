# Shahriyar Mammadli
# Import required libraries
import cv2
import sqlite3
import tkinter
import PIL.Image
import PIL.ImageTk
from tkinter import simpledialog
import time
import helperFunctions as hf

# Detect the face using OpenCV
def detectFace(img):
    # Convert the image to grayscele
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Local Binary Pattern (LBP) is used as face detector. This is a...
    # ...simple, but effective solution.
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    # Detect the faces in the camera
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    # Check if a face founds
    if (len(faces) == 0):
        return None, None
    # One person is expected, thus, ignore others and take the ery first
    (x, y, w, h) = faces[0]
    # Crop and return the face from the image
    return gray[y:y + w, x:x + h], faces[0]

# The function to draw a rectangle which is used to draw a rectangle...
# ...around the face
def drawRectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Draws a text in the image
def drawText(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# Get the maximum label available
def getMaxLabel(conn):
    value = 0
    sql = "SELECT * FROM person"
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    for row in rows:
        label = row[1]
        if int(label) > value:
            value = int(label)
    return value

# Get the correct label for the specified name
def getCorrectLabel(conn, name):
    value = 0
    sql = "SELECT * FROM person"
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    for row in rows:
        target = row[0]
        if target == name:
            value = row[1]
    return value

# The function to identify a person
def predict(orgImg, labelTag, window, canvas, faceRecognizer, names, ts, frame):
    # Make a copy of the original image
    img = orgImg.copy()
    # Detect the face in the image
    face, rect = detectFace(img)
    # Check if there is a face
    if face is not None:
        # Make a prediction using faceRecognizer
        label, uncertainty = faceRecognizer.predict(face)
        confidence = 100 - uncertainty
        labelText = names[label]
        # Check how confident is the classifier, default 40%
        if (confidence >= 40):
            # Put the info into the frame
            labelTag['text'] = labelText + ' | Confidence: ' + str(int(confidence)) + '%'
            drawText(frame, labelText, rect[0], rect[1] - 5)
        # Wait 3 seconds before openning 
        elif time.time() >= ts + 3:
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(orgImg))
            canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
            answer = simpledialog.askstring("New face?", "Your face hasn't been recognized (" + str(
                int(confidence)) + "%). What is your name?",
                                            parent=window)
            # Insert the new image
            if answer is not None and answer != '':
                PIL.Image.fromarray(img).save('toDatabase.jpg')
                ind = 0
                check = False
                # Check if the person exists. For the sake of simplicity...
                # ...name is considered unique here. It means if there are multiple...
                # ...people with the same name, they will be considered as a same person.
                if answer in names:
                    check = True
                    ind = names.index(answer)
                conn = sqlite3.connect('storage.db')
                if check == True:
                    if ind != 0:
                        rightInd = getCorrectLabel(conn, answer)
                        hf.insertPicture(conn, "toDatabase.jpg", int(rightInd), names[int(rightInd)])
                        print("New picture of a existing user has been added.")
                else:
                    index = getMaxLabel(conn) + 1
                    names[index] = answer
                    hf.insertPicture(conn, "toDatabase.jpg", index, names[index])
                    print("New user has been added.")
                conn.close()
            ts = time.time()
    return ts, img