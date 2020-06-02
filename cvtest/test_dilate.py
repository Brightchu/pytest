#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 09:49:11 2020

@author: bright
"""
# from IPython import get_ipython
# get_ipython().magic('reset -sf')
import cv2
import cv2 as cv
import cvbase as cvb
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 300

rgbimg = cv2.imread("Pic_SN001_1_2020-04-03_10-46-43_step2_2546_Success_63_ori.bmp", cv2.IMREAD_COLOR)
img = cv2.imread("Pic_SN001_1_2020-04-03_10-46-43_step2_2546_Success_63_ori.bmp", 0)
# cvb.showimage("fillet", img)

thre,img2 = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
cvb.showimage("binarized", img2)


retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img2)
centroids = centroids.astype(int)
stats[0,4] = 0
maxarea = np.max(stats,axis=0)
maxarea_index = np.argmax(stats, axis=0)

center = (centroids[maxarea_index[4]][0],centroids[maxarea_index[4]][1])

rgbimg = cv.circle(rgbimg, center, 1, (0,0,255), 20)
# cvb.showimage("center", rgbimg)

thre,img2 = cv2.threshold(img,130,255,cv2.THRESH_BINARY)
cvb.showimage("binarized", img2)

dilate_kernel = np.ones((5,5),np.uint8)
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,1))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
dilated = cv2.dilate(img2, dilate_kernel,iterations = 1)
cvb.showimage("dilated", dilated)

cv2.destroyAllWindows()