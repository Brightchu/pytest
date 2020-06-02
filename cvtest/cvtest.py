#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:50:23 2020

@author: bright
"""
import cv2
import cvbase as cvb

img = cv2.imread("/home/bright/Desktop/dvprojects/img/0312/endplane1/Pic_SN001_1_2020-03-12_07-35-24_Success_2_ori.bmp", 0)
cvb.showimage("title", img)