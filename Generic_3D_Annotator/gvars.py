import h5py
from HDF5ImageView import HDF5ImageView
from PyQt5.QtWidgets import QProgressDialog
from numpy import ndarray

imv = None
app = None
win = None
viewer : HDF5ImageView = None
KEY_PROCESSED       : str             = 'processed'
KEY_ORIGINAL        : str             = 'original'
KEY_OBJLIST : str = 'obj_list'
KEY_OBJSTR : str = 'obj'
KEY_NODES : str = 'nodes'
KEY_INTERP : str = 'interpolation'
filepath        : str             = None
open_dir        : str             = None
h5file               : h5py.File       = None
progress_dialog : QProgressDialog = None
cur_obj_name : str = None
objects : dict = dict()
set_axes : bool = False
frame_idcs : ndarray = None