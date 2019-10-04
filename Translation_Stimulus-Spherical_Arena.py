#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author Tim Hladnik

import numpy as np
from numpy import ndarray

# Visualization
from PyQt5 import QtCore, QtWidgets
import pyqtgraph.opengl as gl

import IPython

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def centralCylindrical2DTexture(theta, phi):
    x = theta
    y = np.tan(phi)

    return x, y

def getThetaSubsetIdcs(verts, theta_low: float, theta_high: float) -> ndarray:
    """Returns the vertices which have an theta (azimuth) in between
    a lower and an upper boundary.

    :param theta_low: lower azimuth boundary in degree
    :param theta_high: upper azimuth boundary in degree
    :return:
    """

    thetas, _, _ = cart2sph(verts[:,0], verts[:,1], verts[:,2])

    # Check boundaries
    if theta_low > theta_high:
        Exception('Higher azimuth has to exceed lower azimuth.')

    # Adjust boundaries which exceed [0.0, 360.0]
    while theta_low < 0.0:
        theta_low += 2*np.pi
    while theta_high > 2*np.pi:
        theta_high -= 2*np.pi

    # Filter theta
    bools = None
    if theta_high > theta_low:
        bools = (thetas >= theta_low) & (thetas <= theta_high)
    elif theta_high < theta_low:
        bools = (thetas >= theta_low) | (thetas <= theta_high)
    else:
        Exception('Higher azimuth has to exceed lower azimuth.')

    return bools


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


class MovingStimulus:
    pass


def createTranslationStimulus(verts, v: float = 1., duration: float = 5., frametime: float = .05,
                              direction: str = 'x', pattern: object = None) -> ndarray:
    """Function creates a translation stimulus

    :param sf: spatial frequency of stimulus
    :param v: velocity of stimulus
    :param duration: duration of stimulus in [s]
    :param frametime: time per frame in [ms]
    :return:

    """

    # Calculate azimuth and elevation from cartesian
    #theta, phi, _ = cart2sph(verts[:, 2], verts[:, 0], verts[:, 1])
    #theta, phi, _ = cart2sph(verts[:, 0], verts[:, 1], verts[:, 2])
    theta, phi, _ = cart2sph(verts[:, 1], verts[:, 2], verts[:, 0])
    # Calculate projections on cylinder
    tex_coords = np.array(centralCylindrical2DTexture(theta, phi)).T

    # Set pattern if non was provided
    if pattern is None:
        pattern = Pattern.Bars()

    # Construct stimulus frames
    stimulus = list()
    for t in np.arange(.0, duration, frametime):
        stimulus.append(pattern(tex_coords[:,1], v*t, y=tex_coords[:,0]))

    return np.array(stimulus)

def createSimpleMask(verts, theta_low, theta_high, phi_low, phi_high):
    theta, phi, _ = cart2sph(verts[:,0], verts[:,1], verts[:,2])

    #IPython.embed()

    #mask = getThetaSubsetIdcs(verts, theta_low, theta_high)

    mask = np.zeros(verts.shape[0]).astype(bool)
    #IPython.embed()
    mask[(theta_low < theta) & (theta < theta_high) & (phi_low < phi) & (phi < phi_high)] = True

    return mask


def createHorizontalRectStripeMask(verts, theta_center, phi_center, phi_max):
    theta, phi, _ = cart2sph(verts[:,0], verts[:,1], verts[:,2])

    mask = np.zeros(verts.shape[0]).astype(bool)
    mask[(theta_center-np.pi/2 < theta) & (theta < theta_center+np.pi/2)
         & (phi_center-phi_max/2 < phi) & (phi < phi_center+phi_max/2)] = True

    return mask

def createHorizontalStripeMask(verts, theta_center, phi_center, phi_max):
    theta, phi, _ = cart2sph(verts[:,0], verts[:,1], verts[:,2])

    phi_baseline = -np.sin(theta) * phi_center
    phi_thresh = np.cos(theta-theta_center) * phi_max

    mask = np.zeros(verts.shape[0]).astype(bool)
    #mask[(theta_center-np.pi/2 < theta) & (theta < theta_center+np.pi/2)
    #     & (phi_center-phi_thresh < phi) & (phi < phi_center+phi_thresh)] = True
    mask[(theta_center - np.pi / 2 < theta) & (theta < theta_center + np.pi / 2)
         & (phi_baseline-phi_thresh < phi) & (phi < phi_baseline+phi_thresh)] = True
    return mask


def createFragmentedTranslationStimulus(verts, whole_field, **masked_stimuli):

    # Set whole_field as background
    # TODO: make whole_field optional
    stimulus = whole_field

    for mask_type in masked_stimuli:
        print('Apply mask type "%s"' % mask_type)

        mask = None

        ## Simple masks
        if mask_type == 'left_hemi':
            mask = createSimpleMask(verts, .0, np.pi, -np.pi/2, np.pi/2)

        elif mask_type == 'left_lower_hemi':
            mask = createSimpleMask(verts, .0, np.pi, -np.pi/2, .0)

        elif mask_type == 'right_hemi':
            mask = createSimpleMask(verts, -np.pi, .0, -np.pi/2, .0)

        elif mask_type == 'right_lower_hemi':
            mask = createSimpleMask(verts, -np.pi, .0, -np.pi/2, .0)

        elif mask_type == 'left_horiz_stripe':
            mask = createHorizontalStripeMask(verts, 0., 0., np.pi/4)

        elif mask_type == 'horizontal_stripe':
            mask = createHorizontalStripeMask(verts, -np.pi/2, np.pi/4, np.pi/8)
            #IPython.embed()

        ## Complex masks
        elif isinstance(mask_type, tuple):
            if mask_type[0] == 'horizontal_stripe':
                mask = createHorizontalStripeMask(verts, *mask_type[1])
            elif mask_type[0] == 'vertical_stripe':
                pass

        # Ass masked stimulus to final stimulus
        if mask is not None:
            stimulus[:,mask,:] = masked_stimuli[mask_type][:,mask,:]
        else:
            print('WARNING: no mask for type "%s"' % mask_type)

    return stimulus



if __name__ == '__main__':

    # Create a sphere MeshData
    md = gl.MeshData.sphere(rows=100, cols=200)
    verts = md.vertexes()

    # Create custom pattern
    pattern = Pattern.Bars(sf=2.)

    # Create translation stimulus
    background = createTranslationStimulus(verts, pattern=pattern, duration=20., v=.0)
    foreground = createTranslationStimulus(verts, pattern=pattern, duration=20., v=1.)

    masks = {
        'whole_field': background,
        #'right_lower_hemi' : foreground,
        'horizontal_stripe': foreground
    }
    stimulus = createFragmentedTranslationStimulus(md.vertexes(), **masks)

    # Setup app and window
    app = QtWidgets.QApplication([])
    w = gl.GLViewWidget()
    w.resize(QtCore.QSize(1000, 1000))
    w.show()
    w.setWindowTitle('Stimulus preview')
    w.setCameraPosition(distance=5, azimuth=0)

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
