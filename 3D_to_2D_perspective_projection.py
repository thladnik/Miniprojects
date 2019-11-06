#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author Tim Hladnik

"""
Calculate a 3d to 2d projection of a cube for different camera positions and display distances.
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np


def rotateVertices(vertices, angles):
    """Rotate 3d vertices according to specified Tait–Bryan angles.

    :param vertices:  m x 3 ndarray
    :param angles: size 3 ndarray
    :return: m x 3 rotated vertices

    """
    matRotX = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), np.sin(angles[0])],
        [0, -np.sin(angles[0]), np.cos(angles[0])]
    ])
    matRotY = np.array([
        [np.cos(angles[1]), 0, -np.sin(angles[1])],
        [0, 1, 0],
        [np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    matRotZ = np.array([
        [np.cos(angles[2]), np.sin(angles[2]), 0],
        [-np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])

    # Calculate rotation matrix
    matRot = np.dot(np.dot(matRotX, matRotY), matRotZ)

    # Rotate vertices
    rot_verts = np.dot(matRot, vertices)

    return rot_verts

if __name__ == '__main__':

    # Define vertices
    vertices = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                         [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]])

    # Define faces
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 6], [0, 6, 1],
                      [1, 6, 7], [1, 7, 2], [7, 4, 3], [7, 3, 2], [4, 7, 6], [4, 6, 5]], dtype=np.uint32)
    faces_conn = np.zeros((faces.shape[0], 4), dtype=np.uint32)
    for i in range(faces.shape[0]):
        faces_conn[i, :] = np.append(faces[i, :], faces[i, 0])

    camera_angles = np.array([-np.pi/2, 0., 0.])

    ## Create plots
    fig3d = plt.figure()
    ax3d = fig3d.gca(projection='3d')

    fig2d = dict()
    ax2d = dict()

    # Plot vertices
    ax3d.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'ko', markersize=5, linewidth=1.)
    # Plot faces
    for i in range(faces_conn.shape[0]):
        ax3d.plot(vertices[faces_conn[i, :], 0], vertices[faces_conn[i, :], 1], vertices[faces_conn[i, :], 2], 'k-')

    colors = ['r', 'g', 'y', 'b', 'c']
    markers = ['*', 'o', 'x', '.', '+']
    cam_plotted = list()
    for i, display_distance in enumerate([1.5, 8.]):

        fig2d[display_distance] = plt.figure(num='Display distance %.1f' % display_distance)
        ax2d[display_distance] = fig2d[display_distance].gca()

        for k, camera_position in enumerate([[0., 10., 10.], [0., 10., 2.5], [0., 10., 0.]]):
            camera_position = np.array(camera_position)

            # Camera position (Plot camera point if it hasn't been plotted before
            if k not in cam_plotted:
                ax3d.plot([camera_position[0]], [camera_position[1]], [camera_position[2]], 'k{m}'.format(m=markers[k]),
                          markersize=5, label='Camera {i}'.format(i=k+1))
                cam_plotted.append(k)

            # Display location
            # (For this projection method, positive Z direction is default camera view direction)
            display_pos = np.array([0., 0., display_distance])
            # Plot display location in world coordinates
            plot_disp_pos = rotateVertices(display_pos, camera_angles) + camera_position
            ax3d.plot([plot_disp_pos[0]], [plot_disp_pos[1]], [plot_disp_pos[2]], '{c}{m}'.format(c=colors[i], m=markers[k]), markersize=5)

            # Calculate transform
            rot_verts = rotateVertices(vertices.T - camera_position.reshape((-1, 1)), camera_angles).T

            # Calculate 2d points
            proj_verts = np.zeros((rot_verts.shape[0], 2))
            proj_verts[:, 0] = display_pos[2] / rot_verts[:, 2] * rot_verts[:, 0] + display_pos[0]
            proj_verts[:, 1] = display_pos[2] / rot_verts[:, 2] * rot_verts[:, 1] + display_pos[1]

            # Plot projected 2d points
            ax2d[display_distance].plot(proj_verts[:, 0], proj_verts[:, 1], '{c}{m}'.format(c=colors[i], m=markers[k]), markersize=5)
            # Plot 2d faces
            for j in range(faces_conn.shape[0]):
                ax2d[display_distance].plot(proj_verts[faces_conn[j, :], 0], proj_verts[faces_conn[j, :], 1], '{c}--'.format(c=colors[i]), linewidth=.8)

    ax3d.legend()
    plt.show()
