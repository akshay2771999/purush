
"""
Created on Wed Apr 14 15:08:34 2021

@author: akshay2771999
"""
#I tried different approaches for this problem and this one of them
#I have a taken a reference OK image contour then I have tried to match it with other Not OK images
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from cv2_plt_imshow import cv2_plt_imshow, plt_format
import imutils
import math

#Section 2: Scanning the required file(s)

dir = '/Users/akshay2771999/Desktop/Ok images/OK' #File(s) path
count = 0 #Scanned files count

print('\nDirectory path;', dir, '\n')
for img in os.scandir(dir):
    count += 1
    print('File S.No.',count)
    image1 = cv2.imread(img.path)   #Reading image
    
   # fig = plt.subplots()
    #fig.imshow(image1, extent=[-130,130,0,77])
    #plt.show()
    image = cv2.resize(image1, (1050, 700))   
    crop_1 = image[240:300, 320:660].copy()     #Cropping the copy of the image to the region of interest
   # cv2.imshow("crop_1", crop_1) 
    #cv2.waitKey()  #To visualize if the cropped region is okay
   # cv2_plt_imshow(crop_1)
    crop_2 = image[280:390, 433:550].copy()
    image2=crop_2
   # cv2_plt_imshow(crop_2)
    img_gs = cv2.cvtColor(crop_2, cv2.COLOR_BGR2GRAY)  #convert to grayscale

#inverted binary threshold: 1 for the battery, 0 for the background
    _,thresh = cv2.threshold(img_gs, 250, 1, cv2.THRESH_BINARY_INV)
    #cv2.imshow("crop_1", thresh)


    
#    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    #cv2_plt_imshow(gray)
    blurred = cv2.GaussianBlur(img_gs, (5, 5), 0)    

    edged = cv2.Canny(img_gs, 40, 30)
    #cv2_plt_imshow(edged)

    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = imutils.grab_contours(cnts)
   # c = max(cnts, key=cv2.contourArea
    
    cv2.drawContours(crop_2,cnts,-1, (0, 255, 255), 2)
   # cv2_plt_imshow(crop_2)
    if count >= 1: #Taking one OK image as our reference
        break
    
    
dir = '/Users/akshay2771999/Desktop/NOK Images' #File(s) path
count = 0 #Scanned files count

print('\nDirectory path;', dir, '\n')

#loop for checking Not Ok images
for img in os.scandir(dir):
    count += 1
    #sx= 'file'+ str(count)
    print('File S.No.',count)
    image1 = cv2.imread(img.path)   #Reading image
   # fig = plt.subplots()
    #fig.imshow(image1, extent=[-130,130,0,77])
    #plt.show()
    image = cv2.resize(image1, (1050, 700))   
    crop_1 = image[240:300, 320:660].copy()     #Cropping the copy of the image to the region of interest
   # cv2.imshow("crop_1", crop_1) 
    #cv2.waitKey()  #To visualize if the cropped region is okay
   # cv2_plt_imshow(crop_1)
    crop_3 = image[285:380, 487:585].copy()
    image2=crop_3
    cv2_plt_imshow(crop_2)
    img_gs = cv2.cvtColor(crop_3, cv2.COLOR_BGR2GRAY)  #convert to grayscale

#inverted binary threshold: 1 for the battery, 0 for the background
    _,thresh = cv2.threshold(img_gs, 250, 1, cv2.THRESH_BINARY_INV)
    #cv2.imshow("crop_1", thresh)


    
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    #cv2_plt_imshow(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)    

    edged = cv2.Canny(gray, 20, 53)
   # cv2_plt_imshow(edged)

    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    cnts1 = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts1 = imutils.grab_contours(cnts1)
    #c = max(cnts, key=cv2.contourArea      
    cv2.drawContours(crop_3,cnts1,-1, (0, 0, 255), 2)       
    cv2_plt_imshow(crop_3)
    #matching both contours
    
    ret1 = cv2.matchShapes(cnts[0],cnts1[0],1,0.0) #the function returns a similarity metric (value). A small value indicates that the two shapes are similar and a big value that they are not
    ret2 = cv2.matchShapes(cnts[-1],cnts1[-1],1,0.0) 
    ret=(ret1+ret2)*0.5 #average of both the values
    print(ret)
    if ret>0.99: #minimum similarity metric value after which we can say image is not ok
        print('Image is not OK')
    else:
        print('image is Ok') #if ret<0.13 then image will be considered as Ok

    if count >= 4: #inputting no of images succeding to initial count value that we are planning to scan
        break
    
    