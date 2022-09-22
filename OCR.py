# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 06:53:44 2022

@author: 19001444
"""

from tensorflow.keras.models import load_model
#from imutils.contours import sort_contours
import numpy as np
import argparse
#import  imutils
import cv2
import matplotlib.pyplot as plt
import math
import os

# WordDict = {
    
#     0 : 0 , 1 : 1 , 2 : 2 , 3 : 3 , 4 : 4 , 5 : 5 , 6 : 6 , 7 : 7 , 8 : 8 , 9 : 9 ,
#     10 : 'A' , 11 : 'B' , 12 : 'C' , 13 : 'D' , 14 : 'E' , 15 : 'F' , 16 : 'G' ,
#     17 : 'H' , 18 : 'I' , 19 : 'J' , 20 : 'K' , 21 : 'L' , 22 : 'M' , 23 : 'N' ,
#     24 : 'O' , 25 : 'P' , 26 : 'Q' , 27 : 'R' , 28 : 'S' , 29 : 'T' , 30 : 'U' ,
#     31 : 'V' , 32 : 'W' , 33 : 'X' , 34 : 'Y' , 35 : 'Z'
    
#     }

# model = load_model( 'D:\\UCSC\\3rd year\\1st sem\\3. CG 1\\2. Assigment\\Character Recognition\\Model' )

# import tensorflow as tf
# numberData = tf.keras.datasets.mnist

# ( ( trainData , trainLabels ) , ( testData , testLabels ) ) = numberData.load_data()

# testLabels = tf.keras.utils.to_categorical( testLabels , num_classes = 36 , dtype = 'int64' )

# loss, accuracy = model.evaluate( testData , testLabels )

# import pytesseract

# image = cv2.imread( 'D:\\UCSC\\3rd year\\1st sem\\3. CG 1\\2. Assigment\\Character Recognition\\TestImages\\Untitled.png' )
# gray = cv2.cvtColor( image , cv2.COLOR_BGR2GRAY )
# blurred = cv2.GaussianBlur( gray , ( 5 , 5 ) , 0 )

# edged = cv2.Canny( blurred , 30 , 150 )

# thresh = cv2.threshold( edged , 0 , 255 , cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU )[ 1 ]

# plt.imshow( image )
# plt.show()

# plt.imshow( gray )
# plt.show()

# plt.imshow( blurred )
# plt.show()

# plt.imshow( edged )
# plt.show()

# plt.imshow( thresh )
# plt.show()


# rectKernel = cv2.getStructuringElement( cv2.MORPH_RECT , ( 5 , 5 ) )

# dilation = cv2.dilate( thresh , rectKernel , iterations = 1 )

# contours , hierarchy = cv2.findContours( dilation , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )

# image2 = image.copy()

# for cnt in contours :
    
#     x , y , w , h = cv2.boundingRect( cnt )
    
#     rect = cv2.rectangle( image2 , ( x , y ) , ( x + w , y + h ) , ( 0 , 255 , 0 ) , 2 )
    
#     cropped = image2[ y : y + h , x : x + w ]
    
#     plt.imshow( rect )
#     plt.show()
    
#     plt.imshow( cropped )
#     plt.show()
    
#     print( pytesseract.image_to_string( cropped ) )
    
#convert given image to grayscale imag ( if needed ) and resize to desired height
# def prepareImage( image , height ):
#     assert image.ndim in ( 2 , 3 )
#     if image.ndim == 3:
#         image = cv2.cvtColor( image , cv2.COLOR_BGR2GRAY )
#     h = image.shape[ 0 ]
#     factor = height / h
#     return cv2.resize( image , dsize = None , fx = factor , fy = factor )

#create anisotropic filter kernel according to given parameters
#bluring the image using opencv
# def createKernel( kernelSize , sigma , theta ):
#     assert kernelSize % 2
#     halfSize = kernelSize // 2
    
#     kernel = np.zeros( [ kernelSize , kernelSize ] )
#     sigmaX = sigma
#     sigmaY = sigma + theta
    
#     for i in range( kernelSize ):
#         for j in range( kernelSize ):
#             x = i - halfSize
#             y = j - halfSize
            
#             expTerm = np.exp( -x**2 / ( 2 * sigmaX ) - y**2 / ( 2 * sigmaY ) )
#             xTerm = ( x**2 - sigmaX**2 ) / ( 2 * math.pi * sigmaX**5 * sigmaY )
#             yTerm = ( y**2 - sigmaY**2 ) / ( 2 * math.pi * sigmaY**5 * sigmaX )
            
#             kernel[ i , j ] = ( xTerm + yTerm ) * expTerm
            
#     kernel = kernel / np.sum( kernel )
#     return kernel

# imageDataList = []

# #creating a kernal/filter
# kernel = createKernel( 19 , 9 , 3 )

# #read image
# image = cv2.imread( 'D:\\UCSC\\3rd year\\1st sem\\3. CG 1\\2. Assigment\\Character Recognition\\TestImages\\images.png' )
# plt.figure( figsize = ( 20 , 10 ) )
# plt.imshow( image , cmap = 'gray')

# #prepare image and convert height
# imageGray = prepareImage( image, 2000 )
# plt.figure( figsize = ( 20 , 10 ) )
# plt.imshow( imageGray , cmap = 'gray')

# #apply threshold and filter
# imageFiltered = cv2.filter2D( imageGray , -1 , kernel , borderType = cv2.BORDER_REPLICATE).astype( np.uint8 )
# plt.figure( figsize = ( 20 , 10 ) )
# plt.imshow( imageFiltered , cmap = 'gray')

# ret3 , thresh = cv2.threshold( imageFiltered , 0 , 255 , cv2.THRESH_BINARY | cv2.THRESH_OTSU )
# thresh = 255 - thresh

# plt.figure( figsize = ( 20 , 10 ) )
# plt.imshow( thresh , cmap = 'gray')


# #find contours
# ( componenets, _) = cv2.findContours( thresh , cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE )

# res = []

# for c in componenets:    
    
#     #skip small word candidates
#     if cv2.contourArea( c ) < 0:
#         continue
    
#     currBox = cv2.boundingRect( c )
#     ( x , y , w , h ) = currBox
    
     
#     currImage = image[ y : y + h, x : x + w ]
#     res.append( ( currBox , currImage ) )

# res = sorted( res , key = lambda entry : entry[ 0 ][ 0 ] )
    
# i = 0
# for ( j , w ) in enumerate( res ):
#     ( wordBox , wordImage ) = w
#     ( x , y , w , h ) = wordBox
#     # plt.figure( figsize = ( 20 , 10 ) )
#     plt.imshow( wordImage , cmap = 'gray')
#     i += 1
#     # cv2.imwrite( os.path.join( 'D:\\UCSC\\3rd year\\1st sem\\3. CG 1\\2. Assigment\\Character Recognition\\saveImages' , str( i )+".png" ), wordImage )
#     imageGray = cv2.rectangle( imageGray , ( x , y ), ( x+ w, y + h ), 0 , 1 )
#     # cv2.rectangle( image2 , ( x , y ) , ( x + w , y + h ) , ( 0 , 255 , 0 ) , 2 )

# plt.figure( figsize = ( 20 , 10 ) )
# plt.imshow( imageGray , cmap = 'gray')

imageLocation = 'D:\\UCSC\\3rd year\\1st sem\\3. CG 1\\2. Assigment\\Character Recognition\\TestImages\\F.png'

image = cv2.imread( imageLocation )
gray = cv2.cvtColor( image , cv2.COLOR_BGR2GRAY )
blurred = cv2.GaussianBlur( gray , ( 3 , 3 ) , 0 )
edged = cv2.Canny( blurred , 10 , 100 )

kernel = cv2.getStructuringElement( cv2.MORPH_RECT , ( 3 , 3 ) )

dilate = cv2.dilate( edged , kernel , iterations = 1 )

contours , _ = cv2.findContours( dilate , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )

imageCopy = image.copy()

cv2.drawContours( imageCopy , contours , -1 , ( 0 , 255 , 0 ) , 2 )


imageCopy2 = image.copy()
imageCopy3 = image.copy()
imageCopy4 = image.copy()

x = []
blank = np.full( image.shape[:2] , 255 , dtype = np.uint8 )

print( "Number of contours : " , len( contours ) )

# i = 0
for contour in contours:
    area = cv2.contourArea( contour )
    x,y,w,h = cv2.boundingRect( contour )
    # t = cv2.boundingRect( contour )
    ar = w/float( h )
    # x.append( t )
    
    extractedImage = image[ y : y + h , x : x + w ]
    extractedImage = cv2.cvtColor( extractedImage , cv2.COLOR_BGR2GRAY )
    extractedImage = cv2.resize( extractedImage , ( 28 , 28 ) )
    extractedImage = [ extractedImage ]
    extractedImage = np.array( extractedImage )
    # extractedImage = extractedImage.reshape(  1, 28 , 28 , 1 )
    # print(  extractedImage.shape )
    # cv2.imshow( "ExtractedImage" + str( i ) , extractedImage )
    # i += 1
    # extractedImage = cv2.cvtColor( extractedImage , cv2.COLOR_BGR2GRAY )
    # predictedResults = model.predict( extractedImage )
    
    # if w > 100 and h > 100:
    # if area > 100 and ar < .8:
    cv2.rectangle( image , ( x , y ), ( x + w , y + h ) , ( 0 , 255 , 0 ) , 5 )
    # pred = WordDict[ np.argmax( predictedResults ) ]
    # print( "prediction" , np.argmax( predictedResults ) )
    # print( "string" , str( pred ))
    # cv2.putText( image , str( pred ) , ( x + 10 , y + h + 35 ) , cv2.FONT_HERSHEY_SIMPLEX , 1.2 , ( 0 , 255 , 0 ) , 2 )
    
    # if area > 10 and ar < .8:
    #     cv2.drawContours( imageCopy2 , [contour] , -1 , ( 0 , 0 , 255 ) , 3 )
    #     cv2.drawContours( imageCopy3 , [contour] , -1 , ( 255 , 0 , 0 ) , -1 )
        # x.append( cv2.bitwise_and( image ) )

# print( x )

image = cv2.resize( image , ( 0 , 0 ) , fx = 0.5 , fy = 0.5 )

edged = cv2.resize( edged , ( 0 , 0 ) , fx = 0.5 , fy = 0.5 )

imageCopy = cv2.resize( imageCopy , ( 0 , 0 ) , fx = 0.5 , fy = 0.5 )

imageCopy2 = cv2.resize( imageCopy2 , ( 0 , 0 ) , fx = 0.5 , fy = 0.5 )

imageCopy3 = cv2.resize( imageCopy3 , ( 0 , 0 ) , fx = 0.5 , fy = 0.5 )

blank = cv2.resize( blank , ( 0 , 0 ) , fx = 0.5 , fy = 0.5 )

cv2.imshow( "imageWithRectangle" , image )
# cv2.imshow( "image" , edged )
# cv2.imshow( 'contours' , imageCopy )
# cv2.imshow( "imageCopy2" , imageCopy2 )
# cv2.imshow( "imageCopy3" , imageCopy3 )
# cv2.imshow( "blank" , blank )

cv2.waitKey()
cv2.destroyAllWindows()





