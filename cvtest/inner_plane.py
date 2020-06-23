import cv2
import cvbase as cvb

img = cv2.imread("compensed_ROI.bmp.png", 0)
cvb.showimage("title", img)

