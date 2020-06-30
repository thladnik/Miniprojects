from OpenGL.GL import *
import numpy as np











from PyQt5 import QtCore, QtWidgets

class Window(QtWidgets.QOpenGLWidget):

    def __init__(self):
        QtWidgets.QOpenGLWidget(self)


if __name__ == '__main__':
    app = QtCore.QCoreApplication()