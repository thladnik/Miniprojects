#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author Tim Hladnik

import argparse
import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation

# Visualization
from PyQt5 import QtCore, QtWidgets
import pyqtgraph.opengl as gl

import IPython

gr = 1.61803398874989484820

class IcosahedronSphere:

    corners = [
        [-1, gr, 0],
        [1, gr, 0],
        [-1, -gr, 0],
        [1, -gr, 0],
        [0, -1, gr],
        [0, 1, gr],
        [0, -1, -gr],
        [0, 1, -gr],
        [gr, 0, -1],
        [ gr, 0, 1],
        [-gr, 0, -1],
        [-gr, 0, 1],
    ]

    faces = [

        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],

        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],

        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],

        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    cache = dict()

    def __init__(self, subdiv_lvl):

        # Calculate vertices
        self.vertices = [self.vertex(*v) for v in self.corners]

        # Subdivide faces
        self.subdiv_lvl = subdiv_lvl
        self.subdivide()

    def vertex(self, x, y, z):
        vlen = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return [i/vlen for i in (x, y, z)]

    def midpoint(self, p1, p2):
        key = '%i/%i' % (min(p1, p2), max(p1, p2))

        if key in self.cache:
            return self.cache[key]

        v1 = self.vertices[p1]
        v2 = self.vertices[p2]
        middle = [sum(i)/2 for i in zip(v1, v2)]

        self.vertices.append(self.vertex(*middle))
        index = len(self.vertices) - 1

        self.cache[key] = index

        return index

    def subdivide(self):
        for i in range(self.subdiv_lvl):
            new_faces = []
            for face in self.faces:
                v = [self.midpoint(face[0], face[1]),
                     self.midpoint(face[1], face[2]),
                     self.midpoint(face[2], face[0])]

                new_faces.append([face[0], v[0], v[2]])
                new_faces.append([face[1], v[1], v[0]])
                new_faces.append([face[2], v[2], v[1]])
                new_faces.append([v[0], v[1], v[2]])

            self.faces = new_faces

    def getVertices(self):
        return np.array(self.vertices)

    def getFaces(self):
        return np.array(self.faces)


class Helper:

    @staticmethod
    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    @staticmethod
    def centralCylindrical2DTexture(theta, phi):
        x = theta
        y = np.tan(phi)

        return x, y


class Pattern:

    @staticmethod
    def _createGreyscale(c):
        color = np.ones((c.shape[0], 4))
        color[:, :3] = np.repeat(c[:, np.newaxis], 3, axis=1)
        return color

    @staticmethod
    def Sinusoid(sf: float = 1.):
        def sinusoid(x: ndarray, t: float, **kwargs):
            c = np.cos(1/sf * 2 * np.pi * (x + t))
            return Pattern._createGreyscale(c)

        return sinusoid

    @staticmethod
    def Bars(sf: float = 1.):
        def bars(x: ndarray, shift: float, **kwargs):
            c = np.cos(sf * 2 * np.pi * (x + shift))
            c[c >= .0] = 1.
            c[c < .0] = .0
            return Pattern._createGreyscale(c)

        return bars


class Mask:

    @staticmethod
    def createSimpleMask(verts, theta_low, theta_high, phi_low, phi_high):
        theta, phi, _ = Helper.cart2sph(verts[:, 0], verts[:, 1], verts[:, 2])

        mask = np.zeros(verts.shape[0]).astype(bool)
        mask[(theta_low < theta) & (theta < theta_high)
             & (phi_low < phi) & (phi < phi_high)] = True

        return mask

    @staticmethod
    def createHorizontalRectStripeMask(verts, theta_center, phi_center, phi_range):
        theta, phi, _ = Helper.cart2sph(verts[:, 0], verts[:, 1], verts[:, 2])

        mask = np.zeros(verts.shape[0]).astype(bool)
        mask[(theta_center - np.pi / 2 < theta) & (theta < theta_center + np.pi / 2)
             & (phi_center - phi_range / 2 < phi) & (phi < phi_center + phi_range / 2)] = True

        return mask

    @staticmethod
    def createHorizontalStripeMask(verts, theta_center, phi_center, phi_range):
        raxis = np.array([1.0, 0.0, 0.0])
        r = Rotation.from_rotvec(phi_center * raxis / np.linalg.norm(raxis))
        verts = r.apply(verts)

        theta, phi, _ = Helper.cart2sph(verts[:, 0], verts[:, 1], verts[:, 2])

        phi_thresh = np.cos(theta - theta_center) * phi_range / 2.

        mask = np.zeros(verts.shape[0]).astype(bool)
        mask[(theta_center - np.pi / 2 < theta) & (theta < theta_center + np.pi / 2)
             & (-phi_thresh < phi) & (phi < +phi_thresh)] = True
        return mask


def createTranslationStimulus(verts, v: float = 1., duration: float = 5., frametime: float = .05,
                              pattern: object = None) -> ndarray:
    """Function creates a translation stimulus

    :param sf: spatial frequency of stimulus
    :param v: velocity of stimulus
    :param duration: duration of stimulus in [s]
    :param frametime: time per frame in [ms]
    :return:

    """

    # Calculate azimuth and elevation from cartesian coordinates
    theta, phi, _ = Helper.cart2sph(verts[:, 1], verts[:, 2], verts[:, 0])
    # Calculate central projection on cylinder
    tex_coords = np.array(Helper.centralCylindrical2DTexture(theta, phi)).T

    # Set pattern if non was provided
    if pattern is None:
        pattern = Pattern.Bars()

    # Construct stimulus frames
    stimulus = list()
    for t in np.arange(.0, duration, frametime):
        stimulus.append(pattern(tex_coords[:,1], v*t, y=tex_coords[:,0]))

    return np.array(stimulus)


def createFragmentedTranslationStimulus(verts, whole_field, **masked_stimuli):

    # Set background
    stimulus = whole_field

    for mask_type in masked_stimuli:
        print('Apply mask type "%s"' % mask_type)
        mask_params = masked_stimuli[mask_type]

        mask = None

        ## Simple masks
        if mask_type == 'left_hemi':
            mask = Mask.createSimpleMask(verts, .0, np.pi, -np.pi/2, np.pi/2)

        elif mask_type == 'left_lower_hemi':
            mask = Mask.createSimpleMask(verts, .0, np.pi, -np.pi/2, .0)

        elif mask_type == 'right_hemi':
            mask = Mask.createSimpleMask(verts, -np.pi, .0, -np.pi/2, .0)

        elif mask_type == 'right_lower_hemi':
            mask = Mask.createSimpleMask(verts, -np.pi, .0, -np.pi/2, .0)

        ## Complex masks
        elif mask_type == 'translation_stripe_left':
            mask = np.zeros(verts.shape[0]).astype(bool)
            for m in mask_params[:-1]:
                mask = mask | Mask.createHorizontalStripeMask(verts, np.pi/2, -m[0], m[1])

        elif mask_type == 'translation_stripe_right':
            mask = np.zeros(verts.shape[0]).astype(bool)
            for m in mask_params[:-1]:
                mask = mask | Mask.createHorizontalStripeMask(verts, -np.pi/2, -m[0], m[1])

        elif mask_type == 'translation_stripe_symmetric':
            mask = np.zeros(verts.shape[0]).astype(bool)
            for m in mask_params[:-1]:
                mask = mask | Mask.createHorizontalStripeMask(verts, np.pi/2, -m[0], m[1])
                mask = mask | Mask.createHorizontalStripeMask(verts, -np.pi/2, m[0], m[1])

        ## Additional masks
        elif mask_type[0] == 'vertical_stripe':
            pass

        # Add masked stimulus to final stimulus
        if mask is not None:
            if isinstance(mask_params, tuple):
                stimulus[:,mask,:] = mask_params[-1][:, mask, :]
            else:
                stimulus[:,mask,:] = mask_params[:,mask,:]
        else:
            print('WARNING: no mask for type "%s"' % mask_type)

    return stimulus



if __name__ == '__main__':

    #####
    ## Create sphere
    use_iso = True
    if use_iso:
        print('Using ISO sphere')
        sphere = IcosahedronSphere(5)
        verts = sphere.getVertices()
        md = gl.MeshData()
        md.setVertexes(verts)
        md.setFaces(sphere.getFaces())
    else:
        print('Using UV sphere')
        # Create a sphere MeshData
        md = gl.MeshData.sphere(rows=100, cols=200)
        verts = md.vertexes()


    #####
    ## Create patterns and use them to generate a translation stimulus
    # TODO: this needs to be done automatically (from file or cmd arguments)
    # Create custom pattern
    pattern1 = Pattern.Bars(sf=2.)
    pattern2 = Pattern.Bars(sf=.7)
    # Create translation stimulus
    background = createTranslationStimulus(verts, pattern=pattern2, duration=20., v=.0)
    foreground = createTranslationStimulus(verts, pattern=pattern1, duration=20., v=1.)

    masks = {
        'whole_field': background,
        #'right_lower_hemi' : foreground,
        #'horizontal_stripe': foreground
        #'translation_stripes_upper_left': foreground,
        #'translation_stripes_upper_right': foreground
        'translation_stripe_right': ([np.pi/4, np.pi/8], [-np.pi/4, np.pi/8], foreground),
        #'translation_stripe_symmetric': ([np.pi/3, np.pi/8], foreground)
    }
    stimulus = createFragmentedTranslationStimulus(md.vertexes(), **masks)


    #####
    ## Here follows the visualization part
    # Setup app and window
    app = QtWidgets.QApplication([])
    w = gl.GLViewWidget()
    w.resize(QtCore.QSize(600, 600))
    w.show()
    w.setWindowTitle('Stimulus preview')
    w.setCameraPosition(distance=3, azimuth=0)

    # Create MeshItem
    g = gl.GLGridItem()
    w.addItem(g)
    mi = gl.GLMeshItem(meshdata=md, smooth=True, glOptions='opaque')
    w.addItem(mi)

    # Define update function
    index = 0
    def update():
        global mi, index, stimulus

        if index >= stimulus.shape[0]:
            print('Resetting stimulus')
            index = 0

        s = stimulus[index]
        s[(verts[:,0] > .92) | (verts[:,0] < -.92),:] = np.array([.0, .0, .0, 1.0])

        md.setVertexColors(s)
        mi.meshDataChanged()

        index += 1

    # Set timer for frame update
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    # Start event loop
    QtWidgets.QApplication.instance().exec_()
