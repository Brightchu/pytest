# from numba import jit
import cv2
import numpy as np


# @jit(nopython=True)
def fillColor(colors, stats, num, area_thresh):

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < area_thresh:
            colors[i] = 0
        else:
            colors[i] = 255


# @jit(nopython=True)
def traverseImage(src, labels, colors):

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            label = labels[i, j]
            src[i, j] = colors[label]


# @jit(nopython=True)
def getTBEdgePoints(src, start, end, h, top_pts, bottom_pts):

    for j in range(start, end):
        for i in range(0, h):
            if src[i, j] > 200:
                top_pts[j-start] = i
                break
        for i in range(h - 1, -1, -1):
            if src[i, j] > 200:
                bottom_pts[j-start] = i
                break


# @jit(nopython=True)
def getLREdgePoints(src, start, end, w, left_pts, right_pts):

    for i in range(start, end):
        for j in range(0, w):
            if src[i, j] > 200:
                left_pts[i - start] = j
                break
        for j in range(w - 1, -1, -1):
            if src[i, j] > 200:
                right_pts[i - start] = j
                break


# @jit(nopython=True)
def fillPixelInPolar(src, dst, R, sin_t, cos_t, x0, y0):

    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            x = np.int(x0 + R[i] * cos_t[j])
            y = np.int(y0 + R[i] * sin_t[j])
            if -1 < x < src.shape[1] and -1 < y < src.shape[0]:
                dst[i, j] = src[y, x]

