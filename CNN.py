# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 23:46:59 2017

@author: Arun
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:39:36 2017

@author: Arun
"""

import keras
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV
import cv2
import skimage.measure
from sklearn.svm import SVC
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix



#Intialize CNN


def intialize():
    
    """
    Function Overview:
    1. Create classifier
    2. Add Convolution
    3. Pooling with a stride
    4. Flatten
    4. Fully connected layer
    """
    
    
    classifier = Sequential()
    #1L
    classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3), activation= 'relu'))
    classifier.add(MaxPooling2D(pool_size= (2,2)))
    #2L
    classifier.add(Convolution2D(32,(3,3),activation= 'relu'))
    classifier.add(MaxPooling2D(pool_size= (2,2)))
    
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation= 'relu'))
    classifier.add(Dropout(0.8, input_shape=(128,)))
    classifier.add(Dense(units=25, activation= 'softmax'))
    
    return classifier
    


def compileclassifier(classifier):
    """Function 
    """
    classifier.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics= ['categorical_accuracy'] )
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory('dataset/train',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='categorical')

    test_data = test_datagen.flow_from_directory('dataset/test',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')
    classifier.fit_generator(train_data,
                             steps_per_epoch=8000,
                             epochs=30,
                             validation_data=test_data,validation_steps=2000)
    
    print("Model fitted and the class information is : {}".format(train_data.class_indices))
    
    
    return classifier

def predictionfunction(classifier):
    
    test_image = image.load_img('dataset/single_prediction/4014.jpg', target_size= (64,64))
    test_image = image.img_to_array(test_image)
    test_image= np.expand_dims(test_image,axis=0)
    output= classifier.predict(test_image)
    print(" The result is {}".format(output))
    return output

    

if __name__ == '__main__':
    
    #start by intializing the CNN
    
    classify = intialize()
    
    print("Fitting Model: ")
    
    fitmodel= compileclassifier(classify)
    
    print(" Starting prediction")
    
    op = predictionfunction(fitmodel)
    
    

