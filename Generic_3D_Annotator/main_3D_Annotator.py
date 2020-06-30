
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QLabel
import pyqtgraph as pg
from scipy.interpolate import interp1d
from matplotlib import cm
import numpy as np
import os
import h5py
from IPython import embed

from HDF5ImageView import HDF5ImageView
import gvars
import processing


################################
### File handling

def import_file():
    """Import a new video/image sequence-type file and create mem-mapped file
    """

    fileinfo = QtWidgets.QFileDialog.getOpenFileName(gvars.win, 'Open file...', gvars.open_dir,
                                                     '[Monochrome] Video Files (*.avi; *.mp4);;'
                                                     '[RGB] Video Files (*.avi; *.mp4)')


    if fileinfo == ('',''):
        return

    close_file()

    videopath = fileinfo[0]
    videoformat = fileinfo[1]

    ### Get video file info
    path_parts = videopath.split(os.path.sep)
    gvars.open_dir = os.path.join(videopath[:-1])
    filename = path_parts[-1]
    ext = filename.split('.')[-1]

    ### Set file path and handle
    gvars.filepath = os.path.join(gvars.open_dir, '{}.hdf5'.format(filename[:-(len(ext) + 1)]))
    if os.path.exists(gvars.filepath):
        confirm_dialog = QtWidgets.QMessageBox.question(gvars.win, 'Overwrite file?',
                                                        'This file has already been imported. Do you want to re-import and overwrite?',
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                                                        QtWidgets.QMessageBox.No)

        if confirm_dialog == QtWidgets.QMessageBox.No:
            open_file()
            return
        elif confirm_dialog == QtWidgets.QMessageBox.Cancel:
            return

    ### Open file
    gvars.h5file = h5py.File(gvars.filepath, 'w')

    ### Setup progress dialog
    progress_dialog = QtWidgets.QProgressDialog('Importing file \"{}\"...'.format(filename), 'Cancel', 0, 1, win)
    progress_dialog.setWindowTitle('Import')
    progress_dialog.show()

    ### Include file format options here
    if ext.lower() == 'avi':
        import_avi(videopath, videoformat)
    elif ext.lower() == 'mp4':
        pass
    else:
        close_file()
        progress_dialog.close()
        return

    ### Close progress dialog
    progress_dialog.close()

    ### Get video infos
    gvars.frame_idcs = np.arange(gvars.h5file[gvars.KEY_ORIGINAL].shape[0], dtype=int)

    ### Set video
    setDataset(gvars.KEY_ORIGINAL)


def open_file():

    fileinfo = QtWidgets.QFileDialog.getOpenFileName(gvars.win, 'Open file...', gvars.open_dir,
                                                     'Imported HDF5 (*.hdf5);;')

    if fileinfo == ('',''):
        return

    close_file()

    gvars.filepath = fileinfo[0]

    ### Open file
    gvars.h5file = h5py.File(gvars.filepath, 'a')

    ### Get video infos
    gvars.frame_idcs = np.arange(gvars.h5file[gvars.KEY_ORIGINAL][:].shape[0], dtype=int)

    ### Set video
    setDataset(gvars.KEY_ORIGINAL)
    update_axes_table()


def close_file():
    if not(gvars.h5file is None):
        gvars.h5file.close()
        gvars.h5file = None
        gvars.filepath = None

################################################################
### Import functions

def import_avi(videopath, format):

    import av

    mono = False
    if format.startswith('[Monochrome]'):
        mono = True

    ### Open video file
    vhandle = av.open(videopath)

    ### Import frames
    t_dim = vhandle.streams.video[0].frames

    gvars.progress_dialog.setMaximum(t_dim)
    for i, image in enumerate(vhandle.decode()):

        ### Get image ndarray
        im = np.flipud(np.rot90(np.asarray(image.to_image()).astype(np.uint8), 1))

        ### Convert to monochrome if necessary
        if mono:
            im = im[:,:,0][:,:,np.newaxis]

        x_dim, y_dim, c_dim = im.shape

        gvars.h5file.require_dataset(gvars.KEY_ORIGINAL,
                                     shape=(t_dim, x_dim, y_dim, c_dim),
                                     dtype=np.uint8,
                                     chunks=(1, x_dim, y_dim, c_dim))



        ### Update progress dialog
        gvars.progress_dialog.setValue(i)
        gvars.app.processEvents()
        ### Set frame data
        gvars.h5file[gvars.KEY_ORIGINAL][i, :, :, :] = im[:, :, :]


def setDataset(dsetName):
    if gvars.h5file is None:
        return

    if not(dsetName in gvars.h5file):
        print('WARNING: dataset {} not in file'.format(dsetName))
        return

    if dsetName == gvars.KEY_ORIGINAL:
        gb_display.btn_raw.setStyleSheet('font-weight:bold;')
        gb_display.btn_processed.setStyleSheet('font-weight:normal;')
    else:
        gb_display.btn_processed.setStyleSheet('font-weight:bold;')
        gb_display.btn_raw.setStyleSheet('font-weight:normal;')

    gvars.viewer.setDataset(gvars.h5file[dsetName])
    gvars.viewer.slider.valueChanged.emit(gvars.viewer.slider.value())


################################################################
### Processing


################################################################
### Object handling

def create_object():

    if not(gvars.KEY_OBJLIST in gvars.h5file.attrs):
        gvars.h5file.attrs[gvars.KEY_OBJLIST] = [gvars.KEY_OBJSTR + str(0)]

    else:
        attrs = gvars.h5file.attrs[gvars.KEY_OBJLIST]
        gvars.h5file.attrs[gvars.KEY_OBJLIST] = [*attrs, gvars.KEY_OBJSTR + str(len(attrs))]


    ### Set new object name
    new_obj = gvars.h5file.attrs[gvars.KEY_OBJLIST][-1]

    ### Create nodes and interpolations
    t_dim = gvars.h5file[gvars.KEY_ORIGINAL].shape[0]
    gvars.h5file.create_group(new_obj)
    gvars.h5file[new_obj].create_dataset(gvars.KEY_NODES, data=np.nan * np.ones((t_dim, 2)), dtype=np.float64)
    gvars.h5file[new_obj].create_dataset(gvars.KEY_INTERP, data=np.nan * np.ones((t_dim, 2)), dtype=np.float64)

    ### Set current object name to new object
    print('Created new Object {}'.format(new_obj))
    edit_object(new_obj)

    create_marker(new_obj)


def create_marker(obj_name):

    if obj_name in gvars.objects:
        return
    print('Create marker for object \'{}\''.format(obj_name))

    gvars.objects[obj_name] = dict()

    ## Set color
    rgb = cmap_lut[list(gvars.h5file.attrs[gvars.KEY_OBJLIST]).index(obj_name), :3]

    ### Create dedicated button
    btn = QtWidgets.QPushButton('Object {}'.format(obj_name))
    btn.clicked.connect(lambda: edit_object(obj_name))
    btn.setStyleSheet('background-color: rgb({},{},{})'.format(*rgb))
    gb_objects.layout().addWidget(btn)
    gvars.objects[obj_name]['btn'] = btn

    ### Create dedicated marker
    ## Create marker
    plotItem = pg.PlotDataItem(x=[0], y=[0],
                               symbolBrush=(*rgb, 0,), symbolPen=None, symbol='x', symbolSize=14,
                               name=obj_name)
    gvars.viewer.view.addItem(plotItem)
    gvars.objects[obj_name]['marker'] = plotItem

def edit_object(idx):
    gvars.cur_obj_name = idx

    print('Edit Object {}'.format(gvars.cur_obj_name))

def add_node(ev=None, pos=None):
    print('holla', ev.double(), pos)

    ### If this is not a double-click
    if not(ev.double()):
        return

    ### If this is an axis calibration click:
    if gvars.set_axes:
        return


    if pos is None:

        pos = gvars.viewer.view.mapSceneToView(ev.scenePos())
        x = pos.x()
        y = pos.y()
    else:
        x,y = pos

    ### Set current frame index
    frame_idx = gvars.viewer.slider.value()
    print('Add new node ({},{}) for object \'{}\' in frame {}'.format(x, y, gvars.cur_obj_name, frame_idx))

    gvars.h5file[gvars.cur_obj_name][gvars.KEY_NODES][frame_idx, :] = [x, y]

    ### Get nodes
    nodes = gvars.h5file[gvars.cur_obj_name][gvars.KEY_NODES]
    node_idcs = gvars.frame_idcs[np.isfinite(nodes[:,0]) & np.isfinite(nodes[:,1])]

    ### If interpolation not possible yet:
    if len(node_idcs) < 2:
        gvars.h5file[gvars.cur_obj_name][gvars.KEY_INTERP][frame_idx, :] = [x, y]
        update_pos_marker()
        return

    ### Else:
    ### Interpolate x and y
    xinterp = interp1d(node_idcs, nodes[node_idcs,0], bounds_error=False)
    yinterp = interp1d(node_idcs, nodes[node_idcs,1], bounds_error=False)

    gvars.h5file[gvars.cur_obj_name][gvars.KEY_INTERP][:, 0] = xinterp(gvars.frame_idcs)
    gvars.h5file[gvars.cur_obj_name][gvars.KEY_INTERP][:, 1] = yinterp(gvars.frame_idcs)

    ### Update marker
    update_pos_marker()

def update_pos_marker():

    frame_idx = gvars.viewer.slider.value()

    if not(gvars.KEY_OBJLIST in gvars.h5file.attrs):
        return

    for obj_name in gvars.h5file.attrs[gvars.KEY_OBJLIST]:

        if not(obj_name in gvars.objects):
            create_marker(obj_name)

        ### Position set?
        cur_pos = gvars.h5file[obj_name][gvars.KEY_INTERP][frame_idx]

        ### Get brush
        sym_brush = gvars.objects[obj_name]['marker'].opts['symbolBrush']
        if isinstance(sym_brush, tuple):
            rgb = sym_brush[:3]
        else:
            rgb = sym_brush.color().getRgb()[:3]

        ### No position for this frame: hide marker
        if np.isnan(cur_pos).any():
            gvars.objects[obj_name]['marker'].setSymbolBrush(*rgb, 0)
        ### Else: show marker in correct position
        else:
            gvars.objects[obj_name]['marker'].setData(x=[cur_pos[0]], y=[cur_pos[1]])
            gvars.objects[obj_name]['marker'].setSymbolBrush(*rgb, 255)


################################################################
### Axes calibration

def start_axes_calibration():
    global set_axes, axes_order, gb_calib

    gvars.viewer.slider.setEnabled(False)
    gb_calib.le_axes_status.setText('Set axis {}'.format(axes_order[0]))

    set_axes = True

def set_axis_point(ev):
    global set_axes, axes_order, gb_calib

    if not(ev.double()) or set_axes is False:
        return

    ### Check if position is within limits
    pos = gvars.viewer.view.mapSceneToView(ev.scenePos())
    x = pos.x()
    y = pos.y()

    if x > gvars.h5file[gvars.KEY_ORIGINAL].shape[1] or x < 0.0 or y > gvars.h5file[gvars.KEY_ORIGINAL].shape[2] or y < 0.0:
        print('Position [{}/{}] out of bounds.'.format(x,y))
        return


    ### Create datasets
    if 'axis_calibration_indices' not in gvars.h5file:
        gvars.h5file.create_dataset('axis_calibration_indices',
                                    shape = (0, 1),
                                    maxshape = (None, 1),
                                    dtype = np.uint64)
    fidx_dset = gvars.h5file['axis_calibration_indices']

    if 'axis_calibration_limits' not in gvars.h5file:
        lim_dset = gvars.h5file.create_dataset('axis_calibration_limits',
                                               shape = (0, 4, 2),
                                               maxshape = (None, 4, 2),
                                               dtype = np.float64,
                                               fillvalue = -1.0)
        ### Save calibration order
        lim_dset.attrs['axis_limit_order'] = axes_order
    lim_dset = gvars.h5file['axis_calibration_limits']

    if 'axis_calibration_scale' not in gvars.h5file:
        gvars.h5file.create_dataset('axis_calibration_scale',
                                    shape = (0, 4),
                                    maxshape = (None, 4),
                                    dtype = np.float64,
                                    fillvalue = -1.0)
    scale_dset = gvars.h5file['axis_calibration_scale']

    fidx = gvars.viewer.slider.value()
    # Calibration for new frame
    if not(fidx in fidx_dset[:]):
        print('Start calibration for frame {}'.format(fidx))
        ### Resize datasets and set frame index
        fidx_dset.resize((fidx_dset.shape[0]+1, *fidx_dset.shape[1:]))
        fidx_dset[-1] = fidx
        lim_dset.resize((lim_dset.shape[0]+1, *lim_dset.shape[1:]))
        scale_dset.resize((scale_dset.shape[0]+1, *scale_dset.shape[1:]))

    ### Get current index
    idx = np.where(fidx_dset[:,0] == fidx)[0][0]

    # If all axis limits have been set already: reset them
    if np.sum(lim_dset[idx,:,0] >= 0) == lim_dset[:].shape[1]:
        print('Reset calbration for frame {}'.format(fidx))
        lim_dset[idx,:] = -1.0


    pidx = sum(lim_dset[idx,:,0] >= 0)
    lim_dset[idx,pidx,:] = [x,y]

    ### Set current axis limit
    print('New calibration point {}:{} [{}/{}] added for frame {}'.format(pidx, axes_order[pidx], x, y, fidx))
    update_axes_table()

    ### If last point was calibrated: ask for scale, update all and exit calibration
    if pidx == len(axes_order)-1:
        print('Calibration completed for frame {}'.format(fidx))
        set_axes = False

        gb_calib.le_axes_status.setText('Set scale')
        update_axes_marker()

        dialog = QtWidgets.QDialog(gvars.win)
        dialog.setWindowTitle('Set scale')
        dialog.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
        dialog.setLayout(QtWidgets.QGridLayout())
        dialog.lbl_set = QLabel('Set scale')
        dialog.lbl_set.setStyleSheet('font-weight:bold; text-alignment:center;')
        dialog.layout().addWidget(dialog.lbl_set, 0, 0, 1, 2)
        dialog.xmin = QtWidgets.QDoubleSpinBox()
        dialog.layout().addWidget(QLabel('Xmin'), 1, 0)
        dialog.layout().addWidget(dialog.xmin, 1, 1)
        dialog.xmax = QtWidgets.QDoubleSpinBox()
        dialog.layout().addWidget(QLabel('Xmax'), 2, 0)
        dialog.layout().addWidget(dialog.xmax, 2, 1)
        dialog.ymin = QtWidgets.QDoubleSpinBox()
        dialog.layout().addWidget(QLabel('Ymin'), 3, 0)
        dialog.layout().addWidget(dialog.ymin, 3, 1)
        dialog.ymax = QtWidgets.QDoubleSpinBox()
        dialog.layout().addWidget(QLabel('Ymax'), 4, 0)
        dialog.layout().addWidget(dialog.ymax, 4, 1)
        dialog.btn_submit = QtWidgets.QPushButton('Save')
        dialog.btn_submit.clicked.connect(dialog.accept)
        dialog.layout().addWidget(dialog.btn_submit, 5, 0, 1, 2)

        if not(dialog.exec_() == QtWidgets.QDialog.Accepted):
            raise Exception('No scale for limits set.')

        scale_dset[idx,:] = [dialog.xmin.value(), dialog.xmax.value(), dialog.ymin.value(), dialog.ymax.value()]

        gvars.viewer.slider.setEnabled(True)
        gb_calib.le_axes_status.setText('')
        update_axes_table()
        return

    gb_calib.le_axes_status.setText('Set axis {}'.format(axes_order[pidx + 1]))


def update_axes_table():
    global gb_calib

    gb_calib.axes_table.clear()
    gb_calib.axes_table.setColumnCount(0)
    gb_calib.axes_table.setRowCount(0)
    if not('axis_calibration_limits' in gvars.h5file) or not('axis_calibration_indices' in gvars.h5file):
        return

    findices = gvars.h5file['axis_calibration_indices']
    limits = gvars.h5file['axis_calibration_limits']
    scale = gvars.h5file['axis_calibration_scale']
    labels = limits.attrs['axis_limit_order']

    ### Set table props
    gb_calib.axes_table.setRowCount(limits.shape[1] + 1)
    gb_calib.axes_table.setVerticalHeaderLabels(['Frame #', *labels])
    gb_calib.axes_table.setColumnCount(findices.shape[0])
    gb_calib.axes_table.setHorizontalHeaderLabels([str(i[0]) for i in findices])

    ### Set table data
    for i in range(findices.shape[0]):
        gb_calib.axes_table.setItem(0, i, QtWidgets.QTableWidgetItem(str(findices[i, 0])))
        gb_calib.axes_table.item(0, i)
        for j in range(len(labels)):
            gb_calib.axes_table.setItem(j + 1, i, QtWidgets.QTableWidgetItem('{}: {}'.format(limits[i, j, :], scale[i, j])))


def clear_axes_calibration():
    global axes_markers
    if not('axis_calibration_indices' in gvars.h5file) or not('axis_calibration_limits'):
        return

    answer = QtWidgets.QMessageBox.question(gvars.win, 'Clear calibration', 'Are you sure you want to delete all previous calibrations?',
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)

    if not(answer == QtWidgets.QMessageBox.Yes):
        return

    del gvars.h5file['axis_calibration_indices']
    del gvars.h5file['axis_calibration_limits']

    update_axes_table()

def update_axes_marker():
    global axes_markers

    x_color = (255, 0, 0)
    y_color = (0, 0, 255)
    ### Create markers
    if not(bool(axes_markers)):
        axes_markers['xlims'] = pg.PlotDataItem(x=[], y=[], name='xlims',
                                                symbol='+', symbolBrush=(*x_color,255,), symbolPen=None, symbolSize=20,
                                                pen=pg.mkPen((255,0,0,255), width=2))
        gvars.viewer.view.addItem(axes_markers['xlims'])

        axes_markers['ylims'] = pg.PlotDataItem(x=[], y=[], name='ylims',
                                                symbol='+', symbolBrush=(*y_color,255,), symbolPen=None, symbolSize=20,
                                                pen=pg.mkPen((0,0,255,255), width=2))
        gvars.viewer.view.addItem(axes_markers['ylims'])



    ### Hide markers
    axes_markers['xlims'].setSymbolBrush((*x_color, 0))
    axes_markers['xlims'].setPen((*x_color, 0))
    axes_markers['ylims'].setSymbolBrush((*y_color, 0))
    axes_markers['ylims'].setPen((*y_color, 0))

    if not('axis_calibration_indices' in gvars.h5file) or not('axis_calibration_limits'):
        return

    ### Check if calibration for frame exists (either for this frame or an earlier one -> lower index)
    fidx = gvars.viewer.slider.value()
    fidx_dset = gvars.h5file['axis_calibration_indices']
    if not(np.any(fidx_dset[:,0] <= fidx)):
        return

    ### Get current index
    idx = np.where(fidx_dset[:,0] <= fidx)[0][-1]

    ### Get limits
    limits = gvars.h5file['axis_calibration_limits']
    labels = list(limits.attrs['axis_limit_order'])
    xlims = [limits[idx,labels.index('xmin'),:], limits[idx,labels.index('xmax'),:]]
    ylims = [limits[idx,labels.index('ymin'),:], limits[idx,labels.index('ymax'),:]]

    ### Show markers
    axes_markers['xlims'].setSymbolBrush((*x_color, 255))
    axes_markers['xlims'].setPen((*x_color, 255))
    axes_markers['ylims'].setSymbolBrush((*y_color, 255))
    axes_markers['ylims'].setPen((*y_color, 255))

    ### Set marker coordinates
    axes_markers['xlims'].setData(x=[xlims[0][0], xlims[1][0]], y=[xlims[0][1], xlims[1][1]])
    axes_markers['ylims'].setData(x=[ylims[0][0], ylims[1][0]], y=[ylims[0][1], ylims[1][1]])




################################################################
### Main

if __name__ == '__main__':

    ### Create application
    gvars.app = QtWidgets.QApplication([])

    ### Signals
    sigCalibrationPointSet = QtCore.pyqtSignal(str)

    ### File
    filepath = None
    gvars.h5file = None

    ### Dataset
    d_name = None
    t_dim = None
    x_dim = None
    y_dim = None

    ### Placeholders
    gvars.KEY_ORIGINAL = 'original'
    gvars.KEY_PROCESSED = 'processed'

    gvars.progress_dialog = None

    ### Calibration
    # Axes
    axes_order = ['xmin', 'xmax', 'ymin', 'ymax']
    set_axes = False
    axes_markers = dict()


    gvars.open_dir = './testdata'

    ################################
    ### Setup colormap for markers

    colormap = cm.get_cmap("tab20")
    colormap._init()
    cmap_lut = np.array((colormap._lut * 255))
    cmap_lut = np.append(cmap_lut[::2,:], cmap_lut[1::2,:], axis=0)


    ################################
    ### Setup GUI

    ################
    ### Create window

    gvars.win = QtWidgets.QMainWindow()
    gvars.win.resize(1300,1000)
    gvars.win.setWindowTitle('Generic 3D Annotator')
    cw = QtWidgets.QWidget()
    gvars.win.setCentralWidget(cw)
    cw.setLayout(QtWidgets.QGridLayout())

    ################
    ### Create menu

    mb = QtWidgets.QMenuBar()
    gvars.win.setMenuBar(mb)
    mb_file = mb.addMenu('File')
    mb_file_open = mb_file.addAction('Open file...')
    mb_file_open.setShortcut('Ctrl+O')
    mb_file_open.triggered.connect(open_file)
    mb_file_import = mb_file.addAction('Import image sequence...')
    mb_file_import.setShortcut('Ctrl+I')
    mb_file_import.triggered.connect(import_file)
    mb_process = mb.addMenu('Processing')
    mb_process.act_median_sub = mb_process.addAction('Median subtraction and Normalization')
    mb_process.act_median_sub.triggered.connect(processing.run_median_subtraction)
    mb_process.act_object_detect = mb_process.addAction('Object detection')
    mb_process.act_object_detect.triggered.connect(processing.run_object_detection)

    ################
    ### Create ImageView
    gvars.viewer = HDF5ImageView(gvars.win)
    #gvars.viewer.imv.view.setMouseEnabled(x=False, y=False)
    gvars.viewer.scene.sigMouseClicked.connect(add_node)
    gvars.viewer.scene.sigMouseClicked.connect(set_axis_point)
    cw.layout().addWidget(gvars.viewer, 0, 0)
    gvars.viewer.slider.valueChanged.connect(update_pos_marker)

    ################
    ### Create right panel

    rpanel = QtWidgets.QWidget()
    rpanel.setFixedWidth(500)
    rpanel.setLayout(QtWidgets.QVBoxLayout())
    cw.layout().addWidget(rpanel, 0, 1)

    ########
    ### Display
    gb_display = QtWidgets.QGroupBox('Display')
    gb_display.setLayout(QtWidgets.QVBoxLayout())
    gb_display.btn_raw = QtWidgets.QPushButton('Raw')
    gb_display.btn_raw.clicked.connect(lambda: setDataset(gvars.KEY_ORIGINAL))
    gb_display.layout().addWidget(gb_display.btn_raw)
    gb_display.btn_processed = QtWidgets.QPushButton('Processed')
    gb_display.btn_processed.clicked.connect(lambda: setDataset(gvars.KEY_PROCESSED))
    gb_display.layout().addWidget(gb_display.btn_processed)
    rpanel.layout().addWidget(gb_display)

    ########
    ### Calibration
    ## Axes
    gb_calib = QtWidgets.QGroupBox('Calibration')
    gb_calib.setLayout(QtWidgets.QVBoxLayout())
    # Add label
    gb_calib.lbl_axes = QLabel('Axes for scaling')
    gb_calib.lbl_axes.setStyleSheet('font-weight:bold;')
    gb_calib.layout().addWidget(gb_calib.lbl_axes)
    # Add Button
    gb_calib.btn_add_axes = QtWidgets.QPushButton('')
    gb_calib.btn_add_axes.clicked.connect(start_axes_calibration)
    gvars.viewer.slider.valueChanged.connect(lambda: gb_calib.btn_add_axes.setText('Set axes for frame {}'.format(gvars.viewer.slider.value())))
    gb_calib.layout().addWidget(gb_calib.btn_add_axes)
    # Status indicator
    gb_calib.le_axes_status = QtWidgets.QLineEdit('')
    gb_calib.le_axes_status.setEnabled(False)
    gb_calib.le_axes_status.setStyleSheet('font-weight:bold; color:orange;')
    gb_calib.layout().addWidget(gb_calib.le_axes_status)
    # Table
    gb_calib.axes_table = QtWidgets.QTableWidget()
    gb_calib.axes_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    gb_calib.layout().addWidget(gb_calib.axes_table)
    # Clear button
    gb_calib.btn_clear_axes = QtWidgets.QPushButton('Clear calibration')
    gb_calib.btn_clear_axes.clicked.connect(clear_axes_calibration)
    gb_calib.layout().addWidget(gb_calib.btn_clear_axes)

    # Markers
    gvars.viewer.slider.valueChanged.connect(update_axes_marker)

    ## Reference
    # Label
    gb_calib.lbl_ref = QLabel('References')
    gb_calib.lbl_ref.setStyleSheet('font-weight:bold;')
    gb_calib.layout().addWidget(gb_calib.lbl_ref)
    # Button
    gb_calib.btn_add_ref = QtWidgets.QPushButton('')
    #gb_calib.btn_add_ref.clicked.connect(start_ref_calibration)
    gvars.viewer.slider.valueChanged.connect(lambda: gb_calib.btn_add_ref.setText('Set reference for frame {}'.format(gvars.viewer.slider.value())))
    gb_calib.layout().addWidget(gb_calib.btn_add_ref)



    # Add to centralwidget
    rpanel.layout().addWidget(gb_calib)

    ########
    ### Threshold
    gb_thresh = QtWidgets.QGroupBox('CV2 Threshold')
    gb_thresh.setCheckable(True)
    gb_thresh.setChecked(False)
    gb_thresh.setLayout(QtWidgets.QGridLayout())
    # Threshold
    gb_thresh.thresh = QtWidgets.QSpinBox()
    gb_thresh.thresh.setMinimumWidth(0)
    gb_thresh.thresh.setMaximum(2**8-1)
    gb_thresh.thresh.setValue(120)
    gb_thresh.layout().addWidget(QLabel('Threshold'), 0, 0)
    gb_thresh.layout().addWidget(gb_thresh.thresh, 0, 1)
    # Maxval
    gb_thresh.maxval = QtWidgets.QSpinBox()
    gb_thresh.maxval.setMinimumWidth(0)
    gb_thresh.maxval.setMaximum(2**8-1)
    gb_thresh.maxval.setValue(2**8-1)
    gb_thresh.layout().addWidget(QLabel('Maxval'), 1, 0)
    gb_thresh.layout().addWidget(gb_thresh.maxval, 1, 1)
    # Type
    gb_thresh.threshtype = QtWidgets.QComboBox()
    gb_thresh.threshtype.addItems(['THRESH_BINARY_INV', 'THRESH_BINARY', 'THRESH_TRUNC', 'THRESH_TOZERO_INV', 'THRESH_TOZERO'])
    gb_thresh.layout().addWidget(QLabel('Type'), 2, 0)
    gb_thresh.layout().addWidget(gb_thresh.threshtype, 2, 1)

    ## Adaptive treshold
    gb_thresh.gb_adaptive = QtWidgets.QGroupBox('Adaptive')
    gb_thresh.gb_adaptive.setLayout(QtWidgets.QGridLayout())
    gb_thresh.gb_adaptive.setCheckable(True)
    gb_thresh.gb_adaptive.setChecked(False)
    gb_thresh.gb_adaptive.method = QtWidgets.QComboBox()
    gb_thresh.gb_adaptive.method.addItems(['ADAPTIVE_THRESH_MEAN_C', 'ADAPTIVE_THRESH_GAUSSIAN_C'])
    gb_thresh.gb_adaptive.layout().addWidget(QLabel('Method'), 0, 0)
    gb_thresh.gb_adaptive.layout().addWidget(gb_thresh.gb_adaptive.method, 0, 1)
    gb_thresh.gb_adaptive.block_size = QtWidgets.QSpinBox()
    gb_thresh.gb_adaptive.block_size.setMinimum(3)
    gb_thresh.gb_adaptive.block_size.setMaximum(500)
    gb_thresh.gb_adaptive.block_size.setSingleStep(2)
    gb_thresh.gb_adaptive.block_size.setValue(11)
    gb_thresh.gb_adaptive.layout().addWidget(QLabel('Block size'), 1, 0)
    gb_thresh.gb_adaptive.layout().addWidget(gb_thresh.gb_adaptive.block_size, 1, 1)
    gb_thresh.gb_adaptive.constant = QtWidgets.QSpinBox()
    gb_thresh.gb_adaptive.constant.setMinimum(0)
    gb_thresh.gb_adaptive.constant.setMaximum(2**8-1)
    gb_thresh.gb_adaptive.constant.setValue(5)
    gb_thresh.gb_adaptive.layout().addWidget(QLabel('Constant'), 2, 0)
    gb_thresh.gb_adaptive.layout().addWidget(QLabel('Constant'), 2, 0)
    gb_thresh.gb_adaptive.layout().addWidget(gb_thresh.gb_adaptive.constant, 2, 1)
    # Add adaptive
    gb_thresh.layout().addWidget(gb_thresh.gb_adaptive, 3, 0, 1, 2)

    # Add subpanel
    rpanel.layout().addWidget(gb_thresh)

    ### Update threshold on viewer
    def updateThreshold():
        if gb_thresh.isChecked():
            if gb_thresh.gb_adaptive.isChecked():
                args = [gb_thresh.maxval.value(),
                        gb_thresh.gb_adaptive.method.currentText(),
                        gb_thresh.threshtype.currentText(),
                        gb_thresh.gb_adaptive.block_size.value(),
                        gb_thresh.gb_adaptive.constant.value()]

                gvars.viewer.addImageFilter('threshold', processing.adaptiveThresholdImage, 1, args)
            else:
                args = [gb_thresh.thresh.value(),
                        gb_thresh.maxval.value(),
                        gb_thresh.threshtype.currentText()]

                gvars.viewer.addImageFilter('threshold', processing.thresholdImage, 1, args)

        else:
            gvars.viewer.removeImageFilter('threshold')

    ### Connect events
    # Threshold
    gb_thresh.toggled.connect(updateThreshold)
    gb_thresh.thresh.valueChanged.connect(updateThreshold)
    gb_thresh.maxval.valueChanged.connect(updateThreshold)
    gb_thresh.threshtype.currentTextChanged.connect(updateThreshold)
    # Adaptive
    gb_thresh.gb_adaptive.toggled.connect(updateThreshold)
    gb_thresh.gb_adaptive.method.currentTextChanged.connect(updateThreshold)
    gb_thresh.gb_adaptive.block_size.valueChanged.connect(updateThreshold)
    gb_thresh.gb_adaptive.constant.valueChanged.connect(updateThreshold)

    ########
    ### Control panel
    col2_width = 300
    ### Fish Marker group
    gb_ctrl = QtWidgets.QGroupBox('Control')
    gb_ctrl.setFixedWidth(col2_width)
    gb_ctrl.setLayout(QtWidgets.QVBoxLayout())
    # New object
    btn_new_fish = QtWidgets.QPushButton('New object')
    btn_new_fish.clicked.connect(create_object)
    gb_ctrl.layout().addWidget(btn_new_fish)
    # Apply button
    gb_ctrl.btn_apply = QtWidgets.QPushButton('Apply threshold')

    rpanel.layout().addWidget(gb_ctrl)

    gb_objects = QtWidgets.QGroupBox('Objects')
    gb_objects.setFixedWidth(col2_width)
    gb_objects.setLayout(QtWidgets.QVBoxLayout())
    rpanel.layout().addWidget(gb_objects)

    vSpacer = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
    rpanel.layout().addItem(vSpacer)


    ### Set up window and open/execute
    gvars.win.show()
    gvars.app.exec_()

    close_file()

