#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/aarpit1010/Real-Time-Face-Mask-Detector/blob/master/Face_Mask_Model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


# IMPORTING THE REQUIRED LIBRARIES

import sys
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# uncomment the following line if 'imutils' is not installed in your python kernel
# !{sys.executable} -m pip install imutils
import imutils
from imutils import paths


import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Activation, MaxPooling2D, Flatten
from keras.models import Sequential, load_model
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.regularizers import l2
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

import cv2
import time
import random
import shutil


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# # Let's have a look at our Data

# In[3]:


# Path to the folders containing images

data_path = '/content/drive/My Drive/Colab Notebooks/Face Mask Detector/dataset'
mask_path = '/content/drive/My Drive/Colab Notebooks/Face Mask Detector/train/with_mask/'
nomask_path = '/content/drive/My Drive/Colab Notebooks/Face Mask Detector/train/without_mask/'
test_path = '/content/drive/My Drive/Colab Notebooks/Face Mask Detector/test/'
train_path = '/content/drive/My Drive/Colab Notebooks/Face Mask Detector/train/'


# In[4]:


# function to show images from the input path
def view(path):
    images = list()
    for img in random.sample(os.listdir(path),9):
        images.append(img)
    i = 0
    fig,ax = plt.subplots(nrows=3, ncols=3, figsize=(20,10))
    for row in range(3):
        for col in range(3):
            ax[row,col].imshow(cv2.imread(os.path.join(path,images[i])))
            i+=1


# In[5]:


# sample images of people wearing masks
view(mask_path)


# In[6]:


#sample images of people NOT wearning masks
view(nomask_path)


# # Splitting of Data
# 
# - TRAINING SET
#   - Mask : 658
#   - No Mask : 656
# 
# - TEST SET
#   - Mask : 97
#   - No Mask : 97
# <br><br>
# Since, the dataset is pretty small, image augmentation is performed so as to increase the dataset. We perform Data Augmentation generally to get different varients of the same image without collecting more data which may not be always possible to collect.
# <br><br>
# It is another way to reduce Overfitting on our model, where we increase the amount of training data using information only in our training data and leave the test set untouched.

# # Preparation of Data Pipelining

# In[7]:


batch_size = 32 # Batch Size
epochs = 50  # Number of Epochs
img_size = 224


# In[8]:


# Data Augmentation to increase training dataset size 

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
	      height_shift_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(
        '/content/drive/My Drive/Colab Notebooks/Face Mask Detector/train',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='binary')


test_set = test_datagen.flow_from_directory(
        '/content/drive/My Drive/Colab Notebooks/Face Mask Detector/test',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='binary')


# # Building the Model
# 
# - In the next step, we build our Sequential CNN model with various layers such as Conv2D, MaxPooling2D, Flatten, Dropout and Dense. 
# - In the last Dense layer, we use the ‘**softmax**’ function to output a vector that gives the probability of each of the two classes.
# - Regularization is done to prevent overfitting of the data. It is neccessary since our dataset in not very large and just around 5000 images in total.

# In[9]:


model=Sequential()

model.add(Conv2D(224,(3,3), activation ='relu', input_shape=(img_size,img_size,3), kernel_regularizer=regularizers.l2(0.003)))
model.add(MaxPooling2D() )

model.add(Conv2D(100,(3,3), activation ='relu', kernel_regularizer=regularizers.l2(0.003)))
model.add(MaxPooling2D() )

model.add(Conv2D(100,(3,3), activation ='relu', kernel_regularizer=regularizers.l2(0.003)))
model.add(MaxPooling2D() )

model.add(Conv2D(50,(3,3), activation ='relu', kernel_regularizer=regularizers.l2(0.003)))
model.add(MaxPooling2D() )

model.add(Conv2D(30,(3,3), activation ='relu', kernel_regularizer=regularizers.l2(0.003)))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(90, activation ='relu'))
model.add(Dense(30,  activation = 'relu'))
model.add(Dense(1, activation ='sigmoid'))

model.summary()


# In[10]:


# Optimization of the model is done via Adam optimizer
# Loss is measures in the form of Binary Categorical Cross Entropy as our output contains 2 classes, with_mask and without_mask

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


# In[11]:


#Model Checkpoint to save the model after training, so that it can be re-used while detecting faces


# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "/content/drive/My Drive/Colab Notebooks/Face Mask Detector/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint = ModelCheckpoint(
    filepath = checkpoint_path,
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='auto'
)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Training of the Model is done
history=model.fit(training_set, epochs=epochs, validation_data=test_set)


# In[12]:


# Plotting the loss on validation set w.r.t the number of epochs
plt.plot(history.history['loss'],'r',label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the accuracy on validation set w.r.t the number of epochs
plt.plot(history.history['accuracy'],'r',label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[13]:


# print(model.evaluate(test_data,test_target))


# In[14]:


get_ipython().system('pip install pyyaml h5py  # Required to save models in HDF5 format')


# Now, look at the resulting checkpoints and choose the latest one:

# In[15]:


# Saving the Model trained above, which will be used in future while using Real time data
model.save('/content/drive/My Drive/Colab Notebooks/Face Mask Detector/trained_model.model', history) 
model.save('/content/drive/My Drive/Colab Notebooks/Face Mask Detector/trained_model.h5', history) 


# In[16]:


# IMPLEMENTING LIVE DETECTION OF FACE MASK

# Importing the saved model from the IPython notebook
mymodel=load_model('/content/drive/My Drive/Colab Notebooks/Face Mask Detector/trained_model.h5')

# Importing the Face Classifier XML file containing all features of the face
face_classifier=cv2.CascadeClassifier('/content/drive/My Drive/Colab Notebooks/Face Mask Detector/haarcascade_frontalface_default.xml')

# To open a video via link to be inserted in the () of VideoCapture()
# To open the web cam connected to your laptop/PC, write '0' (without quotes) in the () of VideoCapture()
src_cap=cv2.VideoCapture(0)

while src_cap.isOpened():
    _,img=src_cap.read()

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
        # print(pred)

        if pred[0][0]==1:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.rectangle(img, (x,y-40), (x+w,y), (0,0,255),-1)
            cv2.putText(img,'NO MASK',(250,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
        else:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.rectangle(img, (x,y-40), (x+w,y), (0,255,0),-1)
            cv2.putText(img,'MASK',(250,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
            
        datet=str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        
    # Show the image
    cv2.imshow('LIVE DETECTION',img)
    
    # if key 'q' is press then break out of the loop
    if cv2.waitKey(1)==ord('q'):
        break
    
# Stop video
src_cap.release()

# Close all started windows
cv2.destroyAllWindows()


# In[16]:




