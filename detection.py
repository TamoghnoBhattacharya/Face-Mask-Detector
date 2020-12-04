#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline

import keras
from keras.models import load_model
from keras.models import model_from_json


import cv2
import datetime


# In[2]:


# IMPLEMENTING LIVE DETECTION OF FACE MASK

# Importing the saved model from the IPython notebook
mymodel=load_model('Models/cnn_(2)_model.h5')

# Importing the Face Classifier XML file containing all features of the face
face_classifier=cv2.CascadeClassifier('Cascade Models/haarcascade_frontalface_default.xml')

filename = 'video.avi'
frames_per_second = 24.0
res = '480p'

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='720p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


# To open a video via link to be inserted in the () of VideoCapture()
# To open the web cam connected to your laptop/PC, write '0' (without quotes) in the () of VideoCapture()
cap=cv2.VideoCapture(0)

out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))

# fourcc = cv2.VideoWriter_fourcc(*'XVID') 
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    _,img=cap.read()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detect MultiScale / faces
    faces = face_classifier.detectMultiScale(rgb, 1.3, 5)

    # Draw rectangles around each face
    for (x, y, w, h) in faces:

        #Save just the rectangle faces in SubRecFaces
        face_img = rgb[y:y+w, x:x+w]

        face_img=cv2.resize(face_img,(224,224))
        face_img=face_img/255.0
        face_img=np.reshape(face_img,(224,224,3))
        face_img=np.expand_dims(face_img,axis=0)


        pred=mymodel.predict_classes(face_img)
        print(pred)

        if pred[0]==0:

            cv2.putText(img,'MASK',(250,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
        else:

            cv2.putText(img,'NO MASK',(250,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
            
        datet=str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    
    out.write(img)
    
    # Show the image
    cv2.imshow('LIVE DETECTION',img)
    
    # if key 'q' is press then break out of the loop
    if cv2.waitKey(1)==ord('q'):
        break
    
# Stop video
cap.release()

out.release()

# Close all started windows
cv2.destroyAllWindows()


# In[ ]:




