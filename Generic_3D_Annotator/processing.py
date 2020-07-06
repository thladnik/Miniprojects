import cv2
import numpy as np
from skimage.filters import threshold_otsu

import gv
from median_subtraction import median_norm


def median_norm_filter(image, med_range):
    dshape = gv.f[gv.KEY_ORIGINAL].shape
    i = gv.viewer.slider.value()

    start = i - med_range
    end = i + med_range
    if start < 0:
        end -= start
        start = 0

    if end > dshape[0]:
        start -= end - dshape[0]
        end = dshape[0]

    return median_norm(image, gv.f[gv.KEY_ORIGINAL][start:end, :, :, :])

def threshold_filter(image, thresh, maxval, ttype):
    _, thresh = cv2.threshold(image, thresh, maxval, getattr(cv2, ttype))

    return thresh


def adaptive_threshold_filter(image, maxval, amethod, ttype, block_size, c):
    thresh = cv2.adaptiveThreshold(image, maxval, getattr(cv2, amethod), getattr(cv2, ttype), block_size, c)

    return thresh

def particle_detector(image, percentile):

    ### Separate cell image from background with Otsu thresholding
    cell = image > threshold_otsu(image)

    ### Filter birghtest pixels
    potential_centers = image > np.percentile(image[cell], percentile)

    ### Detect contours
    cnts, hier = cv2.findContours(potential_centers.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ### Reverse sort contour indices by area
    areas = sorted([(cv2.contourArea(cnt), i) for i, cnt in enumerate(cnts)])[::-1]

    ### Filter all contours with > 2 contour points
    cnts2 = [cnts[i] for a, i in areas if a > 0]

    return cnts2

def particle_filter(image, percentile, ):

    cnts = particle_detector(image[:,:,0], percentile)

    cv2.drawContours(image, cnts, -1, (255, 255, 255), 3)

    return image