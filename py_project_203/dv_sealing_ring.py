from dv_base import Basics
import cv2
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing


class SealingRing(Basics):

    def apply(self, image):

        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        start = time.process_time()
        x, y, w, h = (0, 0, 1440, 1430)
        image = image[y:y+h, x:x+w]
        binary_image = self.binariezImage(image, 210, cv2.THRESH_BINARY)
        end = time.process_time()
        print("[INFO] Time cost bin = {}".format(end - start))

        start = time.process_time()
        kernel = np.ones((3, 3), np.uint8)
        erode_image = cv2.erode(binary_image, kernel, iterations=1)
        end = time.process_time()
        print("[INFO] Time cost erode = {}".format(end - start))

        h, w = self.getImageShape(erode_image)

        start = time.process_time()
        self.removeNoise(erode_image, 0.05 * h * w)
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
        projection = self.binariezImage(projection, thresh, cv2.THRESH_BINARY)
        projection = np.uint8(projection)
        self.removeNoise(projection, 20)

        # plt.plot(projection)
        # plt.show()
        boundaries = self.getBoundaries(projection, thresh)

        if len(boundaries) == 6:
            roi = transformed_image[boundaries[2] + 4:boundaries[3] - 4]

            binary_roi1 = self.binariezImage(roi,
                                             max(np.mean(roi) - 70, 0),
                                             cv2.THRESH_BINARY_INV)
            self.removeNoise(binary_roi1, 50)

            binary_roi2 = self.binariezImage(roi,
                                             max(np.mean(roi) - 50, 0),
                                             cv2.THRESH_BINARY_INV)
            self.removeNoise(binary_roi2, 100)

            res = np.vstack((roi, binary_roi1))
            res = np.vstack((res, binary_roi2))
            # self.showImage("res", res)

        return result


if __name__ == "__main__":

    sr = SealingRing()
    fname = list(paths.list_images("/home/xjr/dv_algorithm/data/sealingring/"))

    for i, p in enumerate(fname):
        print("[INFO] Processing {}/{}".format(i+1, len(fname)))
        img = cv2.imread(p, 0)

        if img is not None:
            start = time.process_time()
            res = sr.apply(img)
            end = time.process_time()
            print("[INFO] Time cost = {}".format(end - start))
        else:
            print("[INFO] Emtpy Image")
