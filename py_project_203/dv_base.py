import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from utils import *


class Basics(object):

    def getImageShape(self, src):
        """
        :param src: the input image
        :return: shape of the image
        """
        return src.shape[:2]

    def binariezImage(self, src, thresh, mode):
        """
        :param src: the input image
        :param thresh: thresh for binarizing image
        :param mode: BINARY
        :return: binary image
        """
        _, binary_image = cv2.threshold(src, thresh, 255, mode)
        return binary_image

    def removeNoise(self, src, area_thresh):
        """
        :param src: the input image
        :param area_thresh: thresh for removing small connected components
        :return: an image
        """
        retval, labels, stats, centroid = cv2.connectedComponentsWithStats(src)
        colors = np.ones(retval)
        colors[0] = 0
        fillColor(colors, stats, retval, area_thresh)
        traverseImage(src, labels, colors)
        return src

    def getCenterWithStatistic(self, src):

        h, w = self.getImageShape(src)

        start = h//4
        end = h*3//4
        left_pts = np.zeros(end - start)
        right_pts = np.zeros(end - start)
        getLREdgePoints(src, start, end, w, left_pts, right_pts)

        start = w // 4
        end = w * 3 // 4
        top_pts = np.zeros(end - start)
        bottom_pts = np.zeros(end - start)
        getTBEdgePoints(src, start, end, h, top_pts, bottom_pts)

        x = self.getNumWithMostOccurrence(left_pts, right_pts)
        y = self.getNumWithMostOccurrence(top_pts, bottom_pts)

        return [x, y]

    def getNumWithMostOccurrence(self, pt1s, pt2s):

        points = dict()
        for pt1, pt2 in zip(pt1s, pt2s):
            pt = int((pt1 + pt2) // 2)
            if pt in points:
                points[pt] += 1
            else:
                points[pt] = 1

        for key, value in points.items():
            if value == max(points.values()):
                return key

    def getCenter(self, contours, area_thresh):

        x0, y0 = 0.0, 0.0
        index = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > area_thresh:
                x, y, w, h = cv2.boundingRect(cnt)
                index += 1
                x0 += x + w / 2
                y0 += y + w / 2
        if index > 0:
            x0 /= index
            y0 /= index
        else:
            x0 = 0
            y0 = 0

        return [x0, y0]

    def getOuterEdgePoints(self, src):
        """
        :param src:  the input image -> logical image
        :return: edge points of the outer ring
        """
        pts = []
        h, w = self.getImageShape(src)
        for i in range(w):
            for j in range(h):
                if src[j, i] > 200:
                    pts.append([i, j])
                    break
        return pts

    def getInnerEdgePints(self, src):
        """
        :param src: the input image
        :return: the edge points of the inner ring
        """

        pts = []
        h, w = self.getImageShape(src)
        mid = w // 2
        for i in range(h-1, 0, -1):
            if src[i, mid] < 20:
                for j in range(mid, 0, -1):
                    if src[i, j] > 200:
                        pts.append([j, i])
                        break

                for k in range(mid, w, 1):
                    if src[i, k] > 200:
                        pts.append([k, i])
                        break
            else:
                break
        return pts

    def getContours(self, src):
        contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def getContours(self, src, area_thresh):

        contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > area_thresh:
                new_contours.append(cnt)

        return new_contours

    def getCenterAndRadius(self, pts):
        """
        :param pts:  the edge points
        :return: center and radius
        """

        sum_x = 0.0
        sum_y = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0
        sum_x3 = 0.0
        sum_y3 = 0.0
        sum_xy = 0.0
        sum_x1y2 = 0.0
        sum_x2y1 = 0.0
        for pt in pts:
            x = pt[0]
            y = pt[1]
            x2 = x * x
            y2 = y * y
            sum_x += x
            sum_y += y
            sum_x2 += x2
            sum_y2 += y2
            sum_x3 += x2 * x
            sum_y3 += y2 * y
            sum_xy += x * y
            sum_x1y2 += x * y2
            sum_x2y1 += x2 * y

        N = len(pts)
        C = N * sum_x2 - sum_x * sum_x
        D = N * sum_xy - sum_x * sum_y
        E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x
        G = N * sum_y2 - sum_y * sum_y
        H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y
        a = (H * D - E * G) / (C * G - D * D)
        b = (H * C - E * D) / (D * D - G * C)
        c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N

        center_x = a / (-2)
        center_y = b / (-2)
        radius = np.sqrt(a * a + b * b - 4 * c) / 2

        return [center_x, center_y], radius

    def cart2polar(self, src, center, rMin, rMax, thetaMin, thetaMax):
        """ transform x-y into rho-theta """

        h = int(np.ceil(rMax - rMin))
        w = int(np.ceil(rMax * (thetaMax - thetaMin)))
        dst = np.zeros((h, w), np.uint8)
        r = np.arange(rMin, rMin + h, 1)
        theta = np.arange(0, 2*np.pi, (thetaMax - thetaMin) / w)
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        fillPixelInPolar(src, dst, r, sin_t, cos_t, center[0], center[1])

        return dst

    def getProjectionOnY(self, src):

        return np.mean(src, 1)

    def getProjectionOnX(self, src):

        return np.mean(src, 0)

    def getBoundaries(self, projection, thresh):

        round_flag = False
        boundaries = []
        for i, p in enumerate(projection):

            if not round_flag:
                if p >= thresh:
                    boundaries.append(i)
                    round_flag = True
            else:
                if p <= thresh:
                    boundaries.append(i)
                    round_flag = False

        return boundaries

    def polar2cart(self, point, center, radius, thetaOffset):
        """ transform rho-theta to x-y """

        if thetaOffset > 0:
            x = center[0] + point[1] * np.cos(point[0] / radius + thetaOffset + np.pi)
            y = center[1] + point[1] * np.sin(point[0] / radius + thetaOffset + np.pi)
        else:
            x = center[0] + point[1] * np.cos(point[0] / radius)
            y = center[1] + point[1] * np.sin(point[0] / radius)

        return [y, x]

    def getROI(self, src, lower_bound, upper_bound):
        return src[lower_bound:upper_bound]

    def showImage(self, name, image):

        dst = cv2.resize(image, None, None, fx=0.5, fy=0.5)
        cv2.imshow(name, dst)
        cv2.waitKey(0)

    def showContours(self, src, contours):

        image = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            cv2.rectangle(image, rect, (0, 255, 0), 1)
        self.showImage("image", image)

    def show3DImage(self, src, stride):

        fig = plt.figure()
        ax = Axes3D(fig)
        x = np.arange(0, src.shape[1], 1)
        y = np.arange(0, src.shape[0], 1)
        x, y = np.meshgrid(x, y)
        z = src
        ax.plot_surface(x, y, z, rstride=stride, cstride=stride, cmap='rainbow')
        plt.show()

    def getHist(self, src):

        hist = cv2.calcHist([src], [0], None, [256], [0, 256])
        # hist = hist[0:250]
        normalized_hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX)
        return normalized_hist

    def getWidthOfHistogram(self, hist, threshold):

        """ this is to find the width when give the threshold """
        hist_length = 250
        start = 0
        end = hist_length
        start_stop = False
        end_stop = False
        # width = 0
        while (start < end) and (not start_stop or not end_stop):
            if not start_stop:
                if hist[start] > threshold:
                    start_stop = True
                else:
                    start += 1

            if not end_stop:
                if hist[end] > threshold:
                    end_stop = True
                else:
                    end -= 1

        if start >= end:
            width = 0
            mu = 127
        else:
            width = end - start
            mu = (start + end) / 2

        return width, mu

    def autoThreshforDarkSpotDetection(self, src, std_factor, std_kernel_size, mean_factor, mean_kernel_size, bias):

        roi = cv2.blur(src, (3, 3))
        roi = np.float32(roi)
        roi /= 255.0
        # dst = np.zeros(roi.shape)
        std_m = np.zeros(roi.shape)
        mean_m = np.zeros(roi.shape)

        if std_factor > 0:

            std_kernel = np.ones(std_kernel_size, np.float32)
            n = np.sum(np.sum(std_kernel))

            temp_mean = cv2.filter2D(roi, -1, std_kernel)
            temp_mean = np.multiply(temp_mean, temp_mean) / (n * n)
            variance_m = cv2.filter2D(np.multiply(roi, roi), -1, std_kernel) / n
            temp = np.where(variance_m - temp_mean > np.zeros(roi.shape), variance_m - temp_mean, np.zeros(roi.shape))
            std_m = np.sqrt(temp) * 255.0

        if mean_factor > 0:
            mean_kernel = np.ones(mean_kernel_size, np.float32)
            mean_m = cv2.filter2D(roi, -1, mean_kernel)

        if std_factor is 0 and mean_factor is 0:
            if bias is 0:
                bias = 70
            dst = bias
        else:
            dst = std_m * std_factor + mean_m * mean_factor + bias

        return dst

    def MIN(self, arr, thresh):
        thresh = np.ones(arr.shape) * thresh
        return np.where(arr < thresh, arr, thresh)

    def MAX(self, arr, thresh):
        thresh = np.ones(arr.shape) * thresh
        return np.where(arr > thresh, arr, thresh)

    def binarizeArray(self, arr, thresh_arr, mode):
        ONE = np.ones((arr.shape[0], 1))
        ZERO = np.zeros((arr.shape[0], 1))
        if mode is 'greater_than':
            temp = np.where(arr > thresh_arr, ONE, ZERO)
        elif mode is 'less_than':
            temp = np.where(arr < thresh_arr, ONE, ZERO)
        return temp

    def detectScratching(self, src):
        # src = cv2.blur(src, (5, 5))
        dst = np.zeros(src.shape)
        num_of_cols_as_a_whole = 4
        num_of_traversal = src.shape[1] // num_of_cols_as_a_whole
        print(self.getImageShape(src))

        dark_spot_thresh_lb = 25
        dark_spot_thresh_ub = 70
        dark_spot_coefficient = 0.4

        bright_spot_thresh_lb = 180
        bright_spot_thresh_ub = 230
        bright_spot_coefficient = 2

        for i in range(num_of_traversal):
            temp_ones = np.ones((src.shape[0], num_of_cols_as_a_whole))
            temp_arr = src[:, num_of_cols_as_a_whole*i:num_of_cols_as_a_whole*(i+1)]
            temp_mean = np.mean(temp_arr)

            dark_spot_thresh = self.MIN(self.MAX(temp_mean, dark_spot_thresh_lb), dark_spot_thresh_ub)
            bright_spot_thresh = self.MAX(self.MIN(temp_mean, bright_spot_thresh_ub), bright_spot_thresh_lb)
            dst[:, num_of_cols_as_a_whole*i:num_of_cols_as_a_whole*(i+1)] \
                = cv2.bitwise_or(self.binarizeArray(temp_arr, dark_spot_thresh, 'less_than'),
                                 self.binarizeArray(temp_arr, bright_spot_thresh_ub, 'greater_than'))

        return dst

    def preprocessImage(self, src, binary_thresh, area_thresh, min_r, max_r, min_theta, max_theta):

        print("[INFO] image size = {}".format(self.getImageShape(src)))
        binary_image = self.binariezImage(src, binary_thresh, cv2.THRESH_BINARY)
        re_binary_image = self.removeNoise(binary_image, area_thresh)

        contours = self.getContours(re_binary_image, 0)
        center = self.getCenter(contours, area_thresh)
        print("[INFO] center = {}".format(center))

        transformed_image = self.cart2polar(src, center, min_r, max_r, min_theta, max_theta)
        self.showImage("transformed_image", transformed_image)
        return transformed_image

    def getBoundariesValues(self, src, projection_thresh):

        projection_y = self.getProjectionOnY(src)
        thresh = self.getThresh(projection_y)
        plt.plot(projection_y)
        plt.plot(np.ones(projection_y.shape) * thresh)
        plt.plot(np.ones(projection_y.shape) * thresh * 0.9)
        plt.plot(np.ones(projection_y.shape) * thresh * 0.8)
        plt.plot(np.ones(projection_y.shape) * thresh * 0.7)
        plt.plot(np.ones(projection_y.shape) * thresh * 0.6)
        plt.show()

        boundaries = self.getBoundaries(projection_y, projection_thresh)
        print("[INFO] boundaries = {}".format(boundaries))
        return boundaries

    def getThresh(self, projection):

        avg = np.mean(projection)
        s = 0
        num = 0
        for p in projection:
            if p >= avg:
                s += p
                num += 1
        return s / (num + 0.000000001)

    def scratchingDetection(self, src):

        projection = self.getProjectionOnX(src)
        plt.figure()
        plt.plot(projection, label='projection')
        plt.legend()
        plt.show()

        roi_ds = self.detectScratching(src)
        self.showImage("roi_ds", roi_ds)

    def histogramDetection(self, src, thresh):

        hist = self.getHist(src)
        plt.figure()
        plt.plot(hist, label='hist')
        plt.show()

        hist_width, hist_mu = self.getWidthOfHistogram(hist, thresh)
        print("[INFO] hist_width = {}, hist_mu = {}".format(hist_width, hist_mu))

    def autoThreshDetection(self, src):

        binary_roi = np.ones(src.shape)
        binary_thresh = self.autoThreshforDarkSpotDetection(src, 4, (3, 3), 0.6, (3, 3), 100)
        binary_roi = np.where(src < binary_thresh, binary_roi, binary_roi * 0)
        self.showImage("binary_roi", binary_roi)
        self.show3DImage(binary_thresh, 5)

