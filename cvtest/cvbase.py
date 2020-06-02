#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:20:04 2020

@author: bright
"""
import cv2
# print(cv2.__version__)

def showimage(title, srcimg):
    cv2.namedWindow(title, cv2.WINDOW_FREERATIO)
    cv2.imshow(title,srcimg)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
