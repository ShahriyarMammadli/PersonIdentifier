# PersonIdentifier
PersonIdentifier is a Computer Vision system to detect a person's identity and gender realtime.
## Architecture
The project is built using Tensorflow, Keras, sqlite3, OpenCV, and Tkinter.

- Tkintker is used to build a user interface.
- LBPH face recognizer is used to detect the person.
- Haar-cascade is used to detect the faces from the frame.
- Prebuild gender-detection model is used (["Author"](https://github.com/arunponnusamy))

## How it works
The system can have input from external source, reading from the folder, or from the camera. 

## Usage
For the very first run, you need to provide at least one image for a single person from an external source. External images can be added anytime. To achieve that you need to provide the images n the following format.

* trainingData/
  * "person" + [label] + "-" + [Fullname]/
    * [imageId]
As an example, 2 images for a single person is added to the folder. 
* trainingData/
  * person1-John Travolta/
    * image1 (does not have to in this format, image names can be anything)
    * image2
    
Before importing external images decomment the following line in the personIdentifier.py.
```python
hf.importData(conn, "trainingData")
```

If you do not import the external images then you should comment it again, other wise it will reimport same images.

![Example](/jt.png)

When system does not recognize a person a dialbox appears to ask the name. Consider that Fullname field is considered unique for the sake of simplicity. Thus, if you inputted more than one person with same name they will be considered as a same person.
