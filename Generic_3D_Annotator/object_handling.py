import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d

import gv
import axis_calibration

def create_object():

    if not(gv.KEY_OBJLIST in gv.f.attrs):
        gv.f.attrs[gv.KEY_OBJLIST] = [gv.KEY_OBJSTR + str(0)]

    else:
        attrs = gv.f.attrs[gv.KEY_OBJLIST]
        gv.f.attrs[gv.KEY_OBJLIST] = [*attrs, gv.KEY_OBJSTR + str(len(attrs))]


    ### Set new object name
    new_obj = gv.f.attrs[gv.KEY_OBJLIST][-1]

    ### Create nodes and interpolations
    t_dim = gv.f[gv.KEY_ORIGINAL].shape[0]
    gv.f.create_group(new_obj)
    gv.f[new_obj].create_dataset(gv.KEY_NODES, data=np.nan * np.ones((t_dim, 2)), dtype=np.float64)
    gv.f[new_obj].create_dataset(gv.KEY_NODEINTERP, data=np.nan * np.ones((t_dim, 2)), dtype=np.float64)

    ### Set current object name to new object
    print('Created new Object {}'.format(new_obj))
    edit_object(new_obj)

    create_node_marker(new_obj)


def create_node_marker(obj_name):

    ### If marker already exists: do nothing
    if obj_name in gv.w.viewer.objects:
        return

    print('Create marker for object \'{}\''.format(obj_name))

    gv.w.viewer.objects[obj_name] = dict()

    ## Set color
    rgb = gv.cmap_lut[list(gv.f.attrs[gv.KEY_OBJLIST]).index(obj_name), :3]

    ### Create dedicated button
    btn = QtWidgets.QPushButton('Object {}'.format(obj_name))
    btn.clicked.connect(lambda: edit_object(obj_name))
    btn.setStyleSheet('background-color: rgb({},{},{})'.format(*rgb))
    gv.w.gb_objects.layout().addWidget(btn)
    gv.w.viewer.objects[obj_name]['btn'] = btn

    ### Create dedicated marker
    ## Create marker
    plotItem = pg.PlotDataItem(x=[0], y=[0],
                               symbolBrush=(*rgb, 0,), symbolPen=None, symbol='x', symbolSize=20,
                               name=obj_name)
    gv.w.viewer.view.addItem(plotItem)
    gv.w.viewer.objects[obj_name]['marker'] = plotItem

def edit_object(idx):
    gv.cur_obj_name = idx

    print('Edit Object {}'.format(gv.cur_obj_name))

def add_node(ev=None, pos=None):

    ### If this is not a double-click
    if not(ev.double()):
        return

    ### If this is an axis calibration click:
    if gv.set_axes:
        return


    if pos is None:

        pos = gv.w.viewer.view.mapSceneToView(ev.scenePos())
        x = pos.x()
        y = pos.y()
    else:
        x,y = pos

    ### Set current frame index
    frame_idx = gv.w.viewer.slider.value()
    print('Add new node ({},{}) for object \'{}\' in frame {}'.format(x, y, gv.cur_obj_name, frame_idx))

    gv.f[gv.cur_obj_name][gv.KEY_NODES][frame_idx, :] = [x, y]

    ### Get nodes
    nodes = gv.f[gv.cur_obj_name][gv.KEY_NODES]
    node_idcs = gv.f[gv.KEY_FRAMEIDCS][np.isfinite(nodes[:, 0]) & np.isfinite(nodes[:, 1])]

    ### If interpolation not possible yet:
    if len(node_idcs) < 2:
        gv.f[gv.cur_obj_name][gv.KEY_NODEINTERP][frame_idx, :] = [x, y]
        update_pos_marker()
        return

    ### Else:
    ### Interpolate x and y
    xinterp = interp1d(node_idcs, nodes[node_idcs,0], bounds_error=False)
    yinterp = interp1d(node_idcs, nodes[node_idcs,1], bounds_error=False)

    gv.f[gv.cur_obj_name][gv.KEY_NODEINTERP][:, 0] = xinterp(gv.f[gv.KEY_FRAMEIDCS])
    gv.f[gv.cur_obj_name][gv.KEY_NODEINTERP][:, 1] = yinterp(gv.f[gv.KEY_FRAMEIDCS])

    ### Update marker
    update_pos_marker()

def update_pos_marker():

    if gv.f is None:
        return

    frame_idx = gv.w.viewer.slider.value()

    if not(gv.KEY_OBJLIST in gv.f.attrs):
        return

    for obj_name in gv.f.attrs[gv.KEY_OBJLIST]:

        if not(obj_name in gv.w.viewer.objects):
            create_node_marker(obj_name)

        ### Position set?
        cur_pos = gv.f[obj_name][gv.KEY_NODEINTERP][frame_idx]

        ### Get brush
        sym_brush = gv.w.viewer.objects[obj_name]['marker'].opts['symbolBrush']
        if isinstance(sym_brush, tuple):
            rgb = sym_brush[:3]
        else:
            rgb = sym_brush.color().getRgb()[:3]

        ### No position for this frame: hide marker
        if np.isnan(cur_pos).any():
            gv.w.viewer.objects[obj_name]['marker'].setSymbolBrush(*rgb, 0)
        ### Else: show marker in correct position
        else:
            gv.w.viewer.objects[obj_name]['marker'].setData(x=[cur_pos[0]], y=[cur_pos[1]])
            gv.w.viewer.objects[obj_name]['marker'].setSymbolBrush(*rgb, 255)

