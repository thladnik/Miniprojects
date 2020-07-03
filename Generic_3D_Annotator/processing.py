import cv2

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

def run_object_detection():
    pass