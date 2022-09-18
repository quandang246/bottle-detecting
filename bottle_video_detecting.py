# importing the necessary libraries
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
# Creating a VideoCapture object to read the video
cap = cv.VideoCapture(r'C:\Users\admin\OneDrive\Desktop\2.mp4')
 
 
# Loop until the end of the video
while (cap.isOpened()):
 
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv.resize(frame, (540, 380), fx = 0, fy = 0, interpolation = cv.INTER_CUBIC)
 
    # Display the resulting frame 
    cv.imshow('Frame', frame)
 
    # conversion of BGR to grayscale is necessary to apply this operation
    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)   #Chuyen he mau tu RGB sang HSV
    h, s, v = cv.split(gray)                    #Tach cac channel h, s, v

    # GaussianBlur
    kernel = np.ones((5,5),np.float32)/25
    blur = cv.filter2D(v,-1,kernel) 

    # adaptive thresholding to use different threshold
    # values on different regions of the frame.
    ret,th = cv.threshold(blur,200,255,cv.THRESH_BINARY)
    cv.imshow('Thresh', th)
    
    # Canny Edge Detection
    edges = cv.Canny(th,threshold1=30, threshold2=100)
    cv.imshow('Edges', edges)

    # Finding Contours
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))
    cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv.imshow('Contours', frame)

    # Contours Approximation to circles
    contour_list = []
    for contour in contours:
        approx = cv.approxPolyDP(contour,0.01*cv.arcLength(contour,True),True)
        area = cv.contourArea(contour)
        if ((len(approx) > 8) &(area > 70) ):
            contour_list.append(contour)                                        #phuong thuc append them cac phan tu cho contour_list

    print("Number of bottles found = " + str(len(contour_list)))

    cv.drawContours(frame, contour_list,  -1, (255,0,0), 2)
    cv.imshow('Objects Detected',frame)

    # define q as the exit button
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
 
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv.destroyAllWindows()