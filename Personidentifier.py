# Shahriyar Mammadli
# Import required libraries
import cv2
import numpy as np
import sqlite3
import tkinter
import PIL.Image
import PIL.ImageTk
import time
import genderRecognition as gr
import faceRecognition as fr
import helperFunctions as hf
import os
import traceback


# Initializing the time parameter. This will be used to put...
# ...a break between identification attempts.
ts = time.time()
firstTime = True
# Download pre-trained gender-detection model
model = gr.loadGenderModel()

# Initialize the unique person list
names = [""] * 100

# Import external data
print("Preparing data...")
conn = sqlite3.connect('storage.db')
# Decomment this line when you want to import external data
# hf.importData(conn, "trainingData")
maxVal = 0
faces, labels, names, maxVal = hf.retriveDataFromDatabase(conn, names)
conn.close()
print("Data prepared.")

# Create the LBPH face recognizer
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
# Train the model
faceRecognizer.train(faces, np.array(labels))

cascadePath = "opencv-files/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source.", video_source)
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def getFrame(self):
        ret, frame = self.vid.read()
        if ret:
            # Return a boolean success flag and the current frame converted to BGR
            return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # Label showing name and gender of person
        self.labelName = tkinter.Label(window, text="Ad: NaN")
        self.labelName.pack(anchor=tkinter.CENTER, expand=True)

        self.labelGender = tkinter.Label(window, text="Cinsiyyet: NaN")
        self.labelGender.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 100
        self.update()
        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.getFrame()
        test_img1 = frame.copy()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            global ts, firstTime
            # Add a time gap before first run
            if firstTime:
                ts = ts + 7.5
                firstTime = False
            try:
                ts, predImage = fr.predict(test_img1, self.labelName, self.window, self.canvas,  faceRecognizer, names, ts, frame)
                if predImage is not None:
                    self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
                    self.labelGender['text'] = gr.detectGender(test_img1, model)
            except:
                print(traceback.format_exc())
        self.window.after(self.delay, self.update)

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Live Person Idetifier")
# Delete the toDatabase.jpg file which is used to save images from the...
# ...camera to database
if os.path.isfile("toDatabase.jpg"):
    os.remove("toDatabase.jpg")

