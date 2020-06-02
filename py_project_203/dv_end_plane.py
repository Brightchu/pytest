from dv_base import Basics
import cv2
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing


class EndPlane(Basics):

    # def __init__(self):


    def apply(self, image):

        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        start = time.process_time()
        binary_image = self.binariezImage(image, 120, cv2.THRESH_BINARY)
        end = time.process_time()
        print("[INFO] Time cost bin = {}".format(end - start))

        start = time.process_time()
        kernel = np.ones((2, 2), np.uint8)
        erode_image = cv2.erode(binary_image, kernel, iterations=1)
        end = time.process_time()
        print("[INFO] Time cost erode = {}".format(end - start))

        h, w = self.getImageShape(erode_image)

        start = time.process_time()
        self.removeNoise(erode_image, 0.01 * h * w)
        # self.showImage("erode_image", erode_image)
        end = time.process_time()
        print("[INFO] Time cost remove = {}".format(end - start))

        start = time.process_time()
        center = self.getCenterWithStatistic(erode_image)
        end = time.process_time()
        print("[INFO] Time cost center = {}".format(end - start))

        start = time.process_time()
        min_r, max_r = 290, 700
        min_theta, max_theta = 0, 2 * np.pi

        transformed_image = self.cart2polar(image, center, min_r, max_r, min_theta, max_theta)
        # self.showImage("transformed_image", transformed_image)
        end = time.process_time()
        print("[INFO] Time cost trans = {}".format(end - start))

        projection = np.mean(transformed_image, axis=1)
        # plt.plot(projection)
        thresh = 0.85 * self.getThresh(projection)
        # plt.plot(np.ones(projection.shape) * thresh)
        # plt.show()
        # projection = self.binariezImage(projection, thresh * 0.8, cv2.THRESH_BINARY)
        boundaries = self.getBoundaries(projection, thresh)

        print(boundaries)

        if len(boundaries) == 4:
            inner_roi = transformed_image[boundaries[0] + 3:boundaries[1] - 3]
            outer_roi = transformed_image[boundaries[2] + 3:boundaries[3] - 3]

            binary_inner_roi1 = self.binariezImage(inner_roi,
                                                   max(np.mean(inner_roi) - 70, 0),
                                                   cv2.THRESH_BINARY_INV)
            self.removeNoise(binary_inner_roi1, 50)

            binary_outer_roi1 = self.binariezImage(outer_roi,
                                                   max(np.mean(outer_roi) - 70, 0),
                                                   cv2.THRESH_BINARY_INV)
            self.removeNoise(binary_outer_roi1, 50)

            binary_inner_roi2 = self.binariezImage(inner_roi,
                                                   max(np.mean(inner_roi) - 50, 0),
                                                   cv2.THRESH_BINARY_INV)
            self.removeNoise(binary_inner_roi2, 100)

            binary_outer_roi2 = self.binariezImage(outer_roi,
                                                   max(np.mean(outer_roi) - 50, 0),
                                                   cv2.THRESH_BINARY_INV)
            self.removeNoise(binary_outer_roi2, 100)

            res = np.vstack((inner_roi, binary_inner_roi1))
            res = np.vstack((res, binary_inner_roi2))
            res = np.vstack((res, outer_roi))
            res = np.vstack((res, binary_outer_roi1))
            res = np.vstack((res, binary_outer_roi2))
            # self.showImage("res", res)

        return result


if __name__ == "__main__":

    ep = EndPlane()
    fname = list(paths.list_images("/home/xjr/dv_algorithm/data/ep/"))

    for i, p in enumerate(fname):
        print("[INFO] Processing {}/{}".format(i+1, len(fname)))
        img = cv2.imread(p, 0)

        if img is not None:
            start = time.process_time()
            res = ep.apply(img)
            end = time.process_time()
            print("[INFO] Time cost = {}".format(end - start))
        else:
            print("[INFO] Emtpy Image")
