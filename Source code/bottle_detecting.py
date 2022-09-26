#Khai bao modules
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def main(argv):     #khai bao ham main

    #Tai du lieu anh dau vao
    default_file = r'C:\Users\admin\OneDrive\Desktop\Data_Result\1.0.jpg'
    filename = argv[0] if len(argv) > 0 else default_file       #Kiem tra file rong hay khong
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)     #doc anh mau 
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    #Tach gia tri value (channel v) tu anh
    gray = cv.cvtColor(src, cv.COLOR_BGR2HSV)   #Chuyen he mau tu RGB sang HSV
    h, s, v = cv.split(gray)                    #Tach cac channel h, s, v
    plt.imshow(v, 'gray')
    plt.show()

    #loai bo nhieu bang gaussianBlur
    blur = cv.GaussianBlur(v,(3,3),0)           
    plt.imshow(v, 'gray')
    plt.show()

    #Ap dung nguong Binary voi gia tri nguong thich hop
    ret,th = cv.threshold(blur,175,255,cv.THRESH_BINARY)

    plt.subplot(1,3,1), plt.imshow(blur, 'gray')
    plt.title('Gaussian filtered Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2), plt.hist(blur.ravel(), 256)
    plt.title('Histogram'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3),plt.imshow(th,'gray')
    plt.title('Binary Thresholding'), plt.xticks([]), plt.yticks([])
    plt.show()
    cv.imwrite('BinaryThesholding.jpg', th)
    True

    
    #Su dung phat hien duong vien Canny
    edges = cv.Canny(th,threshold1=30, threshold2=100)
    plt.subplot(1,1,1),plt.imshow(edges,cmap = 'gray')
    plt.title('Canny Edge detection'), plt.xticks([]), plt.yticks([])
    plt.show()
    cv.imwrite('edges.jpg',edges)
    True

    # Finding Contours
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    print("Number of Contours found = " + str(len(contours)))
    
    cv.drawContours(src, contours, -1, (0, 255, 0), 3)
    cv.imshow('Contours', src)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Contours Approximation to circles
    contour_list = []
    for contour in contours:
        approx = cv.approxPolyDP(contour,0.01*cv.arcLength(contour,True),True)
        area = cv.contourArea(contour)
        if ((len(approx) > 8) &(area > 70) ):
            contour_list.append(contour)                                        #phuong thuc append them cac phan tu cho contour_list

    print("Number of bottles found = " + str(len(contour_list)))

    cv.drawContours(src, contour_list,  -1, (255,0,0), 2)
    cv.imshow('Objects Detected',src)
    cv.waitKey(0)

if __name__ == "__main__":      #goi ham main de chay chuong trinh
    main(sys.argv[1:])