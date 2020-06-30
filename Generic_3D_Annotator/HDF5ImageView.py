"""
HDF5ImageView is an image view for displaying memory-mapped image sequences from HDF5 formatted files in pyqtgraph.

Adapted in part from pyqtgraph's ImageView:
> ImageView.py -  Widget for basic image dispay and analysis
> Copyright 2010  Luke Campagnola
> Distributed under MIT/X11 license. See license.txt for more information.


2020 Tim Hladnik
"""

import h5py
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
import time
import numpy as np

class HDF5ImageView(QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.vwdgt = QtWidgets.QWidget(self)
        self.vwdgt.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.vwdgt)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.layout().addWidget(self.slider)
        self.slider.valueChanged.connect(self.updateImage)

        ### Viewbox
        self.view = pg.ViewBox()
        #self.view.setMouseEnabled(False, False)
        self.view.setAspectLocked(True)

        ### Graphics view
        self.graphicsView = pg.GraphicsView(self.vwdgt)
        self.vwdgt.layout().addWidget(self.graphicsView)
        self.graphicsView.setCentralItem(self.view)

        ### Scene
        self.scene = self.graphicsView.scene()

        ### Image item
        self.imageItem = pg.ImageItem()
        self.view.addItem(self.imageItem)

        #self.imageItem.setImage(np.random.randint(10, size=(500, 600)))
        self.playTimer = QtCore.QTimer()
        self.playTimer.timeout.connect(self.timeout)

        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown]
        self.keysPressed = dict()
        self._filters = dict()


    def setDataset(self, dset):
        self.dset = dset
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.dset.shape[0]-1)
        self.updateImage()

    def updateImage(self):
        if self.dset is None:
            return

        filters = [filt for filt in self._filters.items()]

        im = np.fliplr(self.dset[self.slider.value(),:,:,:])
        if bool(filters):
            im = im.copy()

            for _, (filt_ord, filt_fun, filt_args) in sorted(filters):
                im = filt_fun(im, *filt_args)

        self.imageItem.setImage(im)
        #
        #self.imageItem.setImage(np.fliplr(self.dset[self.slider.value(),:,:,:]))

    def addImageFilter(self, filt_name, filt_fun, filt_order, filt_args):
        if filt_name in self._filters:
            self.removeImageFilter(filt_name)

        self._filters[filt_name] = (filt_order, filt_fun, filt_args,)

        self.updateImage()

    def removeImageFilter(self, filt_name):
        if not(filt_name in self._filters):
            return

        del self._filters[filt_name]
        self.updateImage()

    def play(self, rate=None):
        """Begin automatically stepping frames forward at the given rate (in fps).
        This can also be accessed by pressing the spacebar."""
        # print "play:", rate
        if rate is None:
            rate = self.fps
        self.playRate = rate

        if rate == 0:
            self.playTimer.stop()
            return

        self.lastPlayTime = time.time()
        if not self.playTimer.isActive():
            self.playTimer.start(16)

    def timeout(self):
        now = time.time()
        dt = now - self.lastPlayTime
        if dt < 0:
            return

        n = int(self.playRate * dt)
        if n != 0:
            self.lastPlayTime += (float(n)/self.playRate)
            if self.slider.value()+n > self.dset.shape[0]:
                self.play(0)
            self.jumpFrames(n)

    def jumpFrames(self, n):
        """Move video frame ahead n frames (may be negative)"""
        self.slider.setValue(self.slider.value()+n)

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Space:
            if self.playRate == 0:
                self.play()
            else:
                self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_Home:
            self.slider.setValue(0)
            self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_End:
            self.slider.setValue(self.dset.shape[0] - 1)
            self.play(0)
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
        else:
            QtWidgets.QWidget.keyPressEvent(self, ev)

    def keyReleaseEvent(self, ev):
        if ev.key() in [QtCore.Qt.Key_Space, QtCore.Qt.Key_Home, QtCore.Qt.Key_End]:
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except:
                self.keysPressed = {}
            self.evalKeyState()
        else:
            QtWidgets.QWidget.keyReleaseEvent(self, ev)

    def evalKeyState(self):
        if len(self.keysPressed) == 1:
            key = list(self.keysPressed.keys())[0]
            if key == QtCore.Qt.Key_Right:
                self.play(20)
                self.jumpFrames(1)
                self.lastPlayTime = time.time() + 0.2  ## 2ms wait before start
                ## This happens *after* jumpFrames, since it might take longer than 2ms
            elif key == QtCore.Qt.Key_Left:
                self.play(-20)
                self.jumpFrames(-1)
                self.lastPlayTime = time.time() + 0.2
            elif key == QtCore.Qt.Key_Up:
                self.play(-100)
            elif key == QtCore.Qt.Key_Down:
                self.play(100)
            elif key == QtCore.Qt.Key_PageUp:
                self.play(-1000)
            elif key == QtCore.Qt.Key_PageDown:
                self.play(1000)
        else:
            self.play(0)