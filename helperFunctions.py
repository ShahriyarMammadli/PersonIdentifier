# Shahriyar Mammadli
# Import required libraries
import traceback
import sqlite3
import faceRecognition as fr
import os
import cv2
import numpy as np

# The function accepts the folder where the samples for different people...
# ...are stored. Then it detects the faces from the samples and hold...
# ...them in a list and another same-size list holds corresponding labels
def importData(conn, path):
    # Read the folder
    folders = os.listdir(path)
    # Initialize faces and labels lists
    faces = []
    labels = []
    # Iteration through different user folders
    for folderName in folders:
        # It is expected that folder names will be in the format of "person"...
        #  ...and then unique label, then "-[Name]" (e.g. person2-Brad), otherwise...
        #  ...they will be discarded.
        if not folderName.startswith("person"):
            continue
        # Extract the label
        label = int(folderName.split("-")[0].replace("person", ""))
        name = folderName.split("-")[1]
        # Form the path for a person's images
        personPath = path + "/" + folderName
        # Retrieve the image names
        fileNames = os.listdir(personPath)
        # Iterate over the images
        for fileName in fileNames:
            # There may be system files which are starts with '.' (dot)
            if fileName.startswith("."):
                continue
            # Form a path for an image
            imagePath = personPath + "/" + fileName
            # Read the image with the help of OpenCV
            image = cv2.imread(imagePath)
            # Connect to the storage file where the data is kept...
            # ...If there is not such file, then it will be created...
            # ...automatically.
            c = conn.execute("""CREATE TABLE IF NOT EXISTS person(
                                name text,
                                label text,
                                image blob
                            )""")
            insertPicture(conn, imagePath, label, name)
            # Do not close the database since, retriveDataFromDatabase() will use it.
            # conn.close()
            # Display an image window to show the image
            height, width = image.shape[:2]
            cv2.imshow("Training on image...", cv2.resize(image, (height, width)))
            cv2.waitKey(30)
            # Detect the face in the image
            face, rect = fr.detectFace(image)
            # Make sure there is a face
            if face is not None:
                # Put the face and label into corresponding arrays
                faces.append(face)
                labels.append(label)
    cv2.destroyAllWindows()

# This function inserts a new picture into the database
def insertPicture(conn, path, label, name):
    # Handle the possible exceptions
    try:
        with open(path, 'rb') as imageFile:
            img = imageFile.read()
            sql = '''INSERT INTO person (name, label, image) VALUES (?, ?, ?);'''
            conn.execute(sql, [name, label, sqlite3.Binary(img)])
            conn.commit()
    except:
        print(traceback.format_exc())

# Get the all the data in the database (person table)
def retriveDataFromDatabase(conn, names):
    # This is used to have the value of maximum user label...
    # ...thus, for a new user maxValue + 1 will be used...
    # ... as a label
    maxValue = 0
    # Lists to hold faces and labels
    faces = []
    labels = []
    # Retrieve everything in the person table
    sql = "SELECT * FROM person"
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    for row in rows:
        # Extract the data from query response
        rawImage = np.fromstring(row[2], np.uint8)
        image = cv2.imdecode(rawImage, cv2.IMREAD_COLOR)
        label = row[1]
        names[int(label)] = row[0]
        # Compare the maxValue to the new sample's value
        if int(label) > maxValue:
            maxValue = int(label)
        # Display the image
        height, width = image.shape[:2]
        cv2.imshow("Train image", cv2.resize(image, (height, width)))
        cv2.waitKey(30)
        # Detect the face
        face, rect = fr.detectFace(image)
        # Make sure there is a face
        if face is not None:
            # Put the face and label into corresponding arrays
            faces.append(face)
            labels.append(int(label))
    cv2.destroyAllWindows()
    return faces, labels, names, maxValue