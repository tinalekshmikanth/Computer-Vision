#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:45:32 2018

@author: tinapraveen
"""
#import all libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Read images
imgl=cv2.imread('leftimg.jpg',0)
imgr=cv2.imread('rightimg.jpg',0)

# read in images using shape()
# Read Left image
rows_l,cols_l = imgl.shape
# Read Right image
rows_r,cols_r = imgr.shape

# Initialize Variables
start=None
select=(0,0,0,0)
x1=[]
x2=[]


# check points of correspondence

def check_points(x1,x2,R,T):
    for pt_1,pt_2 in zip(x1,x2):
        x1_z = np.dot(R[0, :] - pt_2[0]*R[2, :], T)/np.dot(R[0, :] - pt_2[0]*R[2, :],pt_2)
        pt_1_3d = np.array([pt_1[0] * x1_z, pt_2[0] * x1_z, x1_z])
        pt_2_3d = np.dot(R.T,pt_1_3d) - np.dot(R.T,T)
        
        if pt_1_3d[2] < 0 or pt_2_3d[2] <0 :
            return False
        
    return True

#  fundamental matrix calculation
    # Calculate fundamental matrix using 8 point algorithm
    
def calc_fm(x1,x2):
    mtx = np.zeroes((size,9),np.float32)
    for i in range(n):
        mtx[i] =  [ x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                    x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                    x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
# Use singular Value Decomposition
    U,S,V = linalg.svd(mtx)
    fm = V[-1].reshape(3,3)
    U,S,V = linalg.svd(fm)
    S[2] = 0
    fm= dot(U,dot(diag(S),V))
    print(fm)
    
    return fm 


# Normalize fundamental matrix
    # using 8 point algorithm 

def norm_fm(x1,x2):
    x1 = x1 / x1[2]
    x2 = x2/  x2[2]
    # normalize the vectors by deducting from mean and dividing by standard deviation
    x1_mean = np.mean(x1[:2], axis = 0)
    x2_mean = np.mean(x2[:2], axis = 0)
    x1_sd = np.std(x1[ :2],axis =0)
    x2_sd = np.std(x2[:2],axis = 0)
    
    b = np.sqrt(2) / x1_sd
    c = np.array([[b,0,-b*x1_mean[0]],[0,b,-b*x1_mean[1]],[0,0,1]])
    x1 = np.dot(c,x1)
    
    d = np.sqrt(2)/ x2_sd
    e = np.array([[d,0,-d*x1_mean[0]],[0,d,-d*x1_mean[1]],[0,0,1]])
    x2= np.dot(e,x2)
    # call function to calculate fumndamental matrix
    fm = calc_fm(x1,x2)
    fm = np.dot(c.T,np.dot(fm,e))
    
    return fm/fm[2,2]

def a():
    return none

#Calculate epipole:
def calc_epi(fm):
    U,S,V = linalg.svd(fm)
    e = V[-1]
    return e/e[2]

def m():
    return none  


# Function to draw lines in the image

def draw_lines(imgl,imgr,lines,x1,x2):
    rows,cols =imgl.shape
    # convert left image to grayscale
    imgl = cv2.cvtColor(imgl,cv2.COLOR_GRAY2BGR)
    # Conver right image to grayscale
    imgr = cv2.cvtColor(imgr,cv2.COLOR_GRAY2BGR)
    for rows,pt1,pt2 in zip(lines,x1,x2):
        color = tuple(np.random.randint(0,255,3).tolist())
        ptx0,pty0 = map(int, [0, -rows[2]/rows[1] ])
        ptx1,pty1 = map(int, [cols, -(rows[2]+rows[0]*cols)/rows[1] ])
        imgl = cv2.line(imgl, (ptx0,pty0), (ptx1,pty1), color,1)
        imgl = cv2.circle(imgl,tuple(pt1),5,color,-1)
        imgr = cv2.circle(imgr,tuple(pt2),5,color,-1)
        # Return image with lines drawn with points of correspondence
    return imgl,imgr

# Find the interest points
# Use SIFT()
    
sift = cv2.xfeatures2d.SIFT_create()

i,j = sift.detectAndCompute(imgl,None)
k,l = sift.detectAndCompute(imgr,None)

# Take key points using BFMatcher()
value=[]
BF = cv2.BFMatcher()
match = BF.knnMatch(j,l,k=2)

# generating the features and generating points
for w, (u,v) in enumerate(match):
    if u.distance < 0.8*v.distance:
        value.append(u)
        x2.append(k[u.trainIdx].pt)
        x1.append(i[u.queryIdx].pt)
# Converting x1 to integer data type
x1 = np.int32(x1)
# Converting x2 to integer data type
x2 = np.int32(x2)


lines1 = cv2.computeCorrespondEpilines(x2.reshape(-1,1,2),2,fm)
lines1 = lines1.reshape(-1,3)
final_imgl,final_imgr = draw_lines(imgl,imgr,lines1,x1,x2)

lines2 = cv2.computeCorrespondEpilines(x1.reshape(-1,1,2),1,fm)
lines2 = lines2.reshape(-1,3)
final_imgl1,final_imgr1 = draw_lines(imgr,imgl,lines2,x2,x1)

def x():
    return none

print("Fundamental matrix is :",fm)



#Calculate essential matrix

K = np.float32([-83.33333, 0.00000, 250.00000, 0.00000, -83.33333, 250.0000, 0.00000, 0.00000, 1.00000]).reshape(3,3)
K_inv = np.linalg.inv(K)

em = K.T.dot(fm).dot(K)
# Print Essential Matrix
print("Essential Matrix:",em)


# initialize variables
inliers_2 = []
W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
for i in range(len(x1)):
    # normalize and homogenize the image coordinates
    inliers_1.append(K_inv.dot([x1[i][0], x1[i][1], 1.0]))
    inliers_2.append(K_inv.dot([x2[i][0], x2[i][1], 1.0]))

R = U.dot(W).dot(Vt)
T = U[:, 2]
if not check_points(inliers_1, inliers_2, R, T):

 
    T = - U[:, 2]
    if not check_points(inliers_1, inliers_2, R, T):

        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]

        if not check_points(inliers_1, inliers_2, R, T):

  
            T = - U[:, 2]

# Print Rotation Matrix
print("Rotation Matrix:",R)

# Print Translation Matrix

print("Translation :",T)

# Plot left image
plt.subplot(121),plt.imshow(final_imgl)

#Plot right image
plt.subplot(122),plt.imshow(final_imgl1)















    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        