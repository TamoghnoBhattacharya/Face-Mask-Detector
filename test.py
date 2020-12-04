# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:44:57 2020

@author: Mridul
"""
import os
import sys

import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline

import keras
from keras.models import load_model
from keras.models import model_from_json
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import datetime
# Importing the saved model from the IPython notebook
mymodel=load_model('Models\mobilenetmodel1.h5')

# Importing the Face Classifier XML file containing all features of the face
face_classifier=cv2.CascadeClassifier('Cascade Models\haarcascade_frontalface_default.xml')

def detect(image):
    image = np.array(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #rgb = np.array(rgb, dtype='uint8')

    # detect MultiScale / faces
    faces = face_classifier.detectMultiScale(rgb, 1.3, 5)
    
    color_dict={0:(0,255,0),1:(0,0,255)}
    # Draw rectangles around each face
    for (x, y, w, h) in faces:

        #Save just the rectangle faces in SubRecFaces
        face_img = rgb[y:y+w, x:x+w]

        face_img=cv2.resize(face_img,(224,224))
        face_img=face_img/255.0
        face_img=np.reshape(face_img,(224,224,3))
        face_img=np.expand_dims(face_img,axis=0)
        faces = np.vstack([face_img])
        faces = np.array(faces, dtype="float32")
         
        accuracy=mymodel.predict(face_img)[0][0]
        if  accuracy < 0.75:
            cv2.putText(image,'MASK',(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2) 
            cv2.putText(image,f'Accuracy (%): {(1-accuracy)*100:.2f}',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2) 
        else:
            cv2.putText(image,'NO MASK',(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.putText(image,f'Accuracy (%): {accuracy*100:.2f}',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2) 
            
        datet=str(datetime.datetime.now())
        cv2.putText(image,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    return image


def about():
	st.write(
		'''
		**Haar Cascade** is an object detection algorithm.
		It can be used to detect objects in images or videos. 
		The algorithm has four stages:
			1. Haar Feature Selection 
			2. Creating  Integral Images
			3. Adaboost Training
			4. Cascading Classifiers
Read more :point_right: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid
		''')


def main():
    st.title("Mask Detection App :sunglasses: ")
    st.write("**Using the Haar cascade Classifiers**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":

    	st.write("Go to the About section from the sidebar to learn more about it.")
        
        # You can specify more file types below if you want
    	image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

    	if image_file is not None:

    		image = Image.open(image_file)

    		if st.button("Process"):
                
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
    			result_img = detect(image=image)
    			st.image(result_img, use_column_width = True)

    elif choice == "About":
    	about()




if __name__ == "__main__":
    main()
