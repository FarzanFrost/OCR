# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 14:27:51 2022

@author: 19001444
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
#from keras.optimizers import SGD, Adam
#from keras.callbacks import ReduceLROnPlateau, EarlyStopping
#from keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

location = "D:\\UCSC\\3rd year\\1st sem\\3. CG 1\\2. Assigment\\Character Recognition\\A_Z Handwritten Data\\A_Z Handwritten Data.csv"
alphabetDataWithLabels = pd.read_csv( location )

alphabetData = alphabetDataWithLabels.drop( '0' , axis = 1 )
alphabetDataLabels = alphabetDataWithLabels['0']

print( alphabetData.head() )
print( alphabetData.tail() )

numberData = tf.keras.datasets.mnist

( ( trainData , trainLabels ) , ( testData , testLabels ) ) = numberData.load_data()

numberData = np.vstack( [ trainData,testData ] )
numberDataLabels = np.hstack( [ trainLabels , testLabels ] )

alphabetDataLabels += 10

alphabetData = np.array( alphabetData  )

convertToMatrix = []

for row in alphabetData : 
    convertToMatrix.append( row.reshape( ( 28 , 28 ) ) )
    
convertToMatrix = np.array( convertToMatrix )

Data = np.vstack( [ convertToMatrix , numberData ] )
Labels = np.hstack( [ alphabetDataLabels , numberDataLabels ] )

WordDict = {
    
    0 : 0 , 1 : 1 , 2 : 2 , 3 : 3 , 4 : 4 , 5 : 5 , 6 : 6 , 7 : 7 , 8 : 8 , 9 : 9 ,
    10 : 'A' , 11 : 'B' , 12 : 'C' , 13 : 'D' , 14 : 'E' , 15 : 'F' , 16 : 'G' ,
    17 : 'H' , 18 : 'I' , 19 : 'J' , 20 : 'K' , 21 : 'L' , 22 : 'M' , 23 : 'N' ,
    24 : 'O' , 25 : 'P' , 26 : 'Q' , 27 : 'R' , 28 : 'S' , 29 : 'T' , 30 : 'U' ,
    31 : 'V' , 32 : 'W' , 33 : 'X' , 34 : 'Y' , 35 : 'Z'
    
    }

Labels = tf.keras.utils.to_categorical( Labels , num_classes = 36 , dtype = 'int64' )

xTrain , xTest , yTrain , yTest = train_test_split( Data , Labels )

xTrain = xTrain.reshape( xTrain.shape[0] , xTrain.shape[1] , xTrain.shape[2] , 1 )

xTest = xTest.reshape( xTest.shape[0] , xTest.shape[1] , xTest.shape[2] , 1 )


from keras.preprocessing.image import ImageDataGenerator

trainDatagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=20,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

trainingSet = trainDatagen.flow( xTrain , yTrain )

testDatagen = ImageDataGenerator( rescale = 1./255 )

testSet = testDatagen.flow( xTest , yTest )

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28, 28, 1]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=36, activation='softmax'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
history = cnn.fit(x = trainingSet, validation_data = testSet , epochs = 25)

cnn.summary()

print( "The validation accuracy : " , history.history['val_accuracy'] )
print( "The training accuracy : " , history.history['accuracy'] )
print( "The validation loss : " , history.history['val_loss'] )
print( "The training loss : " , history.history['loss'] )

cnn.save( 'saved_model/my_model' )

cnn.save( 'D:\\UCSC\\3rd year\\1st sem\\3. CG 1\\2. Assigment\\Character Recognition\\Model' )

predictedResults = cnn.predict( xTest[ : 9 ] )
print( xTest.shape )

fig, axes = plt.subplots( 3 , 3 , figsize = ( 8 , 9 ) )
axes = axes.flatten()

for i, ax in enumerate( axes ) : 
    img = np.reshape( xTest[ i ] , ( 28 , 28 ) )
    ax.imshow( img , cmap = 'Greys' )
    pred = WordDict[ np.argmax( predictedResults[i] ) ]
    ax.set_title( "Prediction: " + str( pred ) )
    ax.grid()
















