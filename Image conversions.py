#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:47:33 2018

@author: tinapraveen
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
def grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

image = cv2.imread("/Users/tinapraveen/Desktop/image.jpg")
cv2.imshow('image',image)
cv2.waitKey(1000)
cv2.destroyAllWindows()
var=str(input("Input ShortCut :"))
if var=='i':
    image = cv2.imread("/Users/tinapraveen/Desktop/image.jpg")
    cv2.imshow('image',image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
elif var== 'w':
    cv2.imwrite("/Users/tinapraveen/Desktop/out.jpg",image)
elif var== 'g':
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',image_bw)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
elif var== 'G':
    image_gray = grayConversion(image)
    cv2.imshow('image',image_gray)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
elif var== 'c':
    iteration=1
    if iteration==1:
        b=image.copy()
        b[:,:,1]=0
        b[:,:,2]=0
        cv2.imshow('Blue Channel',b)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        iteration=2
    elif iteration==2:
        g=image.copy()
        g[:,:,0]=0
        g[:,:,2]=0
        cv2.imshow('Green Channel',g)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        iteration=3
    elif iteration==3: 
        r=image.copy()
        r[:,:,0]=0
        r[:,:,1]=0
        cv2.imshow('Red Channel',r)
        cv2.waitKey()
        cv2.destroyAllWindows()
        iteration=1
    elif var== 's':
        image = cv2.imread("/Users/tinapraveen/Desktop/image.jpg")
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(image_bw,(5,5))
        cv2.imshow('Smooth Image',blur)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        