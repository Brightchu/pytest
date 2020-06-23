#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:50:23 2020

@author: bright
"""
import cv2
import cvbase as cvb

img = cv2.imread("compensed_ROI.bmp.png", 0)
cvb.showimage("title", img)

