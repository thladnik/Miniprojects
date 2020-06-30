import cv2
import numpy as np
from PyQt5 import QtWidgets

import gvars
import median_subtraction

def rotate(dir):
    global imv, meta, data

    gvars.h5file['im_seq'][:].reshape()
    gvars.h5file['im_seq'][:] = np.rot90(gvars.h5file['im_seq'][:], k=dir, axes=(1, 2))

    if not ('rotation' in meta):
        meta['rotation'] = 0
    meta['rotation'] += dir

    ### Set image
    imv.setImage(gvars.h5file[gvars.KEY_ORIGINAL][:])


def run_median_subtraction():

    gvars.h5file.require_dataset(gvars.KEY_PROCESSED,
                                 shape=gvars.h5file[gvars.KEY_ORIGINAL].shape,
                                 dtype=np.uint8,
                                 chunks=(1, *gvars.h5file[gvars.KEY_ORIGINAL].shape[1:]))

    dialog_process = QtWidgets.QProgressDialog('Processing video...', 'Cancel', 0, 1, gvars.win)
    dialog_process.setWindowTitle('Processing')
    dialog_process.setValue(0)

    gvars.h5file[gvars.KEY_PROCESSED][:] = median_subtraction.run(gvars.h5file[gvars.KEY_ORIGINAL][:], report_handle=dialog_process, app=app)


def thresholdImage(image, thresh, maxval, ttype):
    _, thresh = cv2.threshold(image, thresh, maxval, getattr(cv2, ttype))

    return thresh


def adaptiveThresholdImage(image, maxval, amethod, ttype, block_size, c):
    thresh = cv2.adaptiveThreshold(image, maxval, getattr(cv2, amethod), getattr(cv2, ttype), block_size, c)

    return thresh

def run_object_detection():
    pass