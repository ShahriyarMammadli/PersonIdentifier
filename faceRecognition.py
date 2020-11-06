# Shahriyar Mammadli
# Import required libraries
import cv2
import os
import numpy as np
import sqlite3
import tkinter
import PIL.Image
import PIL.ImageTk
import traceback
from tkinter import simpledialog
import time
import helperFunctions as hf

# function to detect face using OpenCV
<<<<<<< HEAD
def detectFace(img):
=======
def detect_face(img):
>>>>>>> 038f78e86069300f87e6849d8cf357ba4cb1cfd6
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]


# function to draw rectangle on image
# according to given (x, y) coordinates and
# given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def getMaxLabel(conn):
    vall = 0
    sql = "SELECT * FROM person"
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    for row in rows:
        label = row[1]
        if int(label) > vall:
            vall = int(label)

    return vall
def getCorrectLabel(conn, name):
    vall = 0
    sql = "SELECT * FROM person"
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    for row in rows:
        target = row[0]
        if target == name:
            vall = row[1]

    return vall

# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the
# subject
def predict(test_img, lbl_name, window, canvas, face_recognizer, names, ts):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)
    if face is not None:
        # predict the image using our face recognizer
        label, confidence = face_recognizer.predict(face)
        label_text = names[label]
<<<<<<< HEAD
=======

        # print(confidence)
>>>>>>> 038f78e86069300f87e6849d8cf357ba4cb1cfd6
        # draw a rectangle around face detected
        draw_rectangle(img, rect)
        # draw name of predicted person
        if (confidence <= 50):
            # get name of respective label returned by face recognizer
            lbl_name['text'] = label_text + ' | Confidence: ' + str(int(100 - confidence)) + '%'
            # print(label_text)
            draw_text(img, label_text, rect[0], rect[1] - 5)
        elif time.time() >= ts + 3:
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(test_img))
            canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
            answer = simpledialog.askstring("New face?", "Your face hasn't been recognized (" + str(
                int(100 - confidence)) + "%). What is your name?",
                                            parent=window)
<<<<<<< HEAD
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
                        print("ADDING WHEN FOUND")
                        rightInd = getCorrectLabel(conn, answer)
                        hf.insertPicture(conn, "toDatabase.jpg", int(rightInd), names[int(rightInd)])
                        print("added WHEN FOUND")
                else:
                    print("addING WHEN NOT FOUND")
                    index = getMaxLabel(conn) + 1
                    names[index] = answer
                    hf.insertPicture(conn, "toDatabase.jpg", index, names[index])
                    print("added WHEN NOT FOUND")
                conn.close()
=======

            if answer is not None:
                if answer != '':
                    ind = 0  # '' 1 1 1
                    check = False
                    for name in names:
                        if name == answer:
                            check = True
                            break
                        if name != "":
                            ind += 1

                    conn = sqlite3.connect('storage.db')
                    if check == True:
                        if ind != 0:
                            print("ADDING WHEN FOUND")
                            rightInd = getCorrectLabel(conn, answer)
                            hf.insertPicture(conn, "toDatabase.jpg", int(rightInd), names)
                            print("added WHEN FOUND")
                    else:
                        print("addING WHEN NOT FOUND")
                        index = getMaxLabel(conn) + 1
                        names[index] = answer
                        hf.insertPicture(conn, "toDatabase.jpg", index, names)
                        print("added WHEN NOT FOUND")
                    conn.close()

                    ind = 0
>>>>>>> 038f78e86069300f87e6849d8cf357ba4cb1cfd6
            ts = time.time()
    return ts, img


