#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author Tim Hladnik

Dev = False

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation
from scipy.io import savemat
import sys

# Visualization
from PyQt5 import QtCore, QtWidgets
import pyqtgraph.opengl as gl

if Dev:
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
    def _createGreyscaleRGBA(c):
        color = np.ones((c.shape[0], 4))
        color[:, :3] = np.repeat(c[:, np.newaxis], 3, axis=1)
        return color

    ## Translational
    @staticmethod
    def Sinusoid(sf: float = 1.):
        def sinusoid(x: ndarray, shift: float, return_rgba: bool=True, **kwargs):
            c = np.cos(180 * sf * np.pi * x + shift)
            if return_rgba:
                return Pattern._createGreyscaleRGBA(c)
            return c

        return sinusoid

    @staticmethod
    def Bars(sf: float = 1.):
        def bars(x: ndarray, shift: float, return_rgba=True, **kwargs):
            c = Pattern.Sinusoid(sf)(x, shift, return_rgba=False, **kwargs)
            c[c >= .0] = 1.
            c[c < .0] = .0
            if return_rgba:
                return Pattern._createGreyscaleRGBA(c)
            return c
        return bars

    ## Rotational
    # TODO: try if rotational stimuli work like this.
    # It's probably better though to rotate the sphere
    # instead of trying to emulate rotation on the 2d canvas.
    #   -> This would require an extra "createRotationStimulus" function similar to "createTranslationStimulus"
    #      which rotates the vertices and recomputes the tex_coords for each frame


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

    :param verts: vertices that make up the sphere (2d ndarray)
    :param sf: approx. LOWEST spatial frequency of stimulus in [cyc/deg]
    :param v: approx. LOWEST velocity of stimulus in [deg/s]
    :param duration: duration of stimulus in [s]
    :param frametime: time per frame in [ms]
    :return:
      stimulus frames as a whole_field representation on the sphere (3d ndarray)

    """

    # Calculate azimuth and elevation from cartesian coordinates
    theta, phi, _ = Helper.cart2sph(verts[:, 1], verts[:, 2], verts[:, 0])
    # Calculate central projection on cylinder
    tex_coords = np.array(Helper.centralCylindrical2DTexture(theta, phi)).T

    # Set pattern if none was provided
    if pattern is None:
        pattern = Pattern.Bars()

    # Construct stimulus frames
    stimulus = list()
    for t in np.arange(.0, duration, frametime):
        stimulus.append(pattern(tex_coords[:,1], v*t, y=tex_coords[:,0]))

    # Return stimulus frames
    return np.array(stimulus)


def applyMasks(verts, whole_field, *masked_stimuli):
    """Applies a set of predefined masks with the specified stimuli.

      Simple masks do not take any mask parameters. Possible types so far are:
        > left_hemi         : left hemisphere
        > left_lower_hemi   : lower left hemisphere
        > right_hemi        : right hemisphere
        > right_lower_hemi  : lower right hemisphere

      Complex masks do take mask parameters. Possible types so far are:
        > transl_stripe_left   : an oval-like mask spanning horizontally from front to back
        > transl_stripe_right  : an oval-like mask spanning horizontally from front to back
        > transl_stripes_symm  : two oval-like masks spanning horizontally from front to back

    :param verts: vertices that make up the sphere (2d ndarray)
    :param whole_field: (background) whole field stimulus frames (3d ndarray)
    :param masked_stimuli: list of mask_types, mask parameters and corresponding stimulus frames
     used to construct the final stimulus frames
    :return:
      final stimulus frames (3d ndarray)
    """

    # Set background
    stimulus = whole_field

    # Overwrite background with masked stimuli
    for newmask in masked_stimuli:
        mask_type = newmask[0]
        newstim = newmask[-1]

        print('Apply mask type <%s>' % mask_type)
        if len(newmask) > 2:
            mask_params = newmask[1:-1]

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
        elif mask_type == 'transl_stripe_left':
            mask = Mask.createHorizontalStripeMask(verts, np.pi/2, -mask_params[0], mask_params[1])

        elif mask_type == 'transl_stripe_right':
            mask = Mask.createHorizontalStripeMask(verts, -np.pi/2, mask_params[0], mask_params[1])

        elif mask_type == 'transl_stripe_symm':
            mask = Mask.createHorizontalStripeMask(verts, np.pi/2, -mask_params[0], mask_params[1])
            mask = mask | Mask.createHorizontalStripeMask(verts, -np.pi/2, mask_params[0], mask_params[1])

        ## Additional masks
        elif mask_type[0] == 'vertical_stripe':
            pass

        # Add new masked stimulus to final stimulus
        if mask is not None:
            stimulus[:,mask,:] = newstim[:,mask,:]
        else:
            print('WARNING: no mask of type <%s>' % mask_type)

    # Before returning stimulus: blank front and back poles
    for fIdx in range(stimulus.shape[0]):
        stimulus[fIdx,(verts[:, 0] > .95) | (verts[:, 0] < -.95),:] = np.array([.0, .0, .0, 1.0])

    # Return final stimulus
    return stimulus


class Stimulus:

    def __init__(self, use_iso = True):

        ## Create sphere vertices
        if use_iso:
            print('Using ISO sphere')
            sphere = IcosahedronSphere(5)
            self.verts = sphere.getVertices()
            self.faces = sphere.getFaces()
        else:
            print('Using UV sphere')
            md = gl.MeshData.sphere(rows=100, cols=200)
            self.verts = md.vertexes()
            self.faces = md.faces()

        self.phases = list()
        self.data = None


    def addPhase(self, phase):
        self.phases.append(phase)


    def compile(self):
        self.data = np.concatenate(self.phases, axis=0)


    def saveAs(self, filename, ext='mat'):
        self.compile()

        composed = dict(vertices=self.verts, stimulus=self.data[:,:,0])  # only save greyscale
        if ext == 'mat':
            savemat('%s.%s' % (filename, ext), composed)


    def display(self, frametime):
        self.compile()

        # Set MeshData
        md = gl.MeshData()
        md.setVertexes(self.verts)
        md.setFaces(self.faces)

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
        self.dispIdx = 0
        def update():

            if self.dispIdx == self.data.shape[0]:
                print('Stimulus presentation finished.')
                self.dispIdx = -1
                timer.stop()
                w.close()

            md.setVertexColors(self.data[self.dispIdx])
            mi.meshDataChanged()

            self.dispIdx += 1

        # Set timer for frame update
        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(1000*frametime)

        # Start event loop
        QtWidgets.QApplication.instance().exec_()


if __name__ == '__main__':

    if 'example01' in sys.argv:

        frametime = .05
        dur = 5.

        stim = Stimulus()

        # Create a pattern
        pattern = Pattern.Bars(sf=2.5/180)

        # Create translation stimuli
        for v in np.linspace(0.5, 5., 10):
            background = createTranslationStimulus(stim.verts,
                                                   pattern=pattern, duration=dur, v=.0, frametime=frametime)
            pos_transl = createTranslationStimulus(stim.verts,
                                                   pattern=pattern, duration=dur, v=v, frametime=frametime)
            neg_transl = createTranslationStimulus(stim.verts,
                                                   pattern=pattern, duration=dur, v=-v, frametime=frametime)

            phase = applyMasks(stim.verts, background,
                               ['transl_stripe_symm', np.pi / 4, np.pi / 4, pos_transl],
                               ['transl_stripe_symm', -np.pi / 4, np.pi / 4, neg_transl],
                               )
            stim.addPhase(phase)

        stim.display(frametime)

        stim.saveAs('example01')

    elif 'example02' in sys.argv:

        frametime = .05
        dur = 5.

        stim = Stimulus()


        # Create translation stimuli
        for sf in np.linspace(1./180, 5./180, 5):
            # Create a pattern
            pattern = Pattern.Bars(sf=sf)
            background = createTranslationStimulus(stim.verts,
                                                   pattern=pattern, duration=dur, v=.0, frametime=frametime)
            translation = createTranslationStimulus(stim.verts,
                                                   pattern=pattern, duration=dur, v=.5, frametime=frametime)

            phase = applyMasks(stim.verts, background,
                               ['left_hemi', translation],
                               )
            stim.addPhase(phase)

        stim.display(frametime)

        stim.saveAs('example02')


    elif 'example03' in sys.argv:

        frametime = .05
        dur = 5.

        stim = Stimulus()

        # Create a pattern
        pattern1 = Pattern.Bars(sf=1./180)
        pattern2 = Pattern.Bars(sf=3./180)

        # Create translation stimuli
        for v in np.linspace(0.5, 3., 10):
            background = createTranslationStimulus(stim.verts,
                                                   pattern=pattern1, duration=dur, v=.0, frametime=frametime)
            translation = createTranslationStimulus(stim.verts,
                                                   pattern=pattern2, duration=dur, v=v, frametime=frametime)

            phase = applyMasks(stim.verts, background,
                               ['right_lower_hemi', translation],
                               )
            stim.addPhase(phase)

        stim.display(frametime)

        stim.saveAs('example03')
