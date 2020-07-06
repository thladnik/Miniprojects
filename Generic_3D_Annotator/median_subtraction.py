import ctypes
import h5py
from multiprocessing import cpu_count, RawArray, Pool, Manager
import numpy as np
import time

import gv

def run():
    print('Run Median subtraction + Range normalization')

    ### Create dataset if necessary
    gv.f.require_dataset(gv.KEY_PROCESSED,
                         shape=gv.f[gv.KEY_ORIGINAL].shape,
                         dtype=np.uint8,
                         chunks=(1, *gv.f[gv.KEY_ORIGINAL].shape[1:]))
    t,x,y,c = gv.f[gv.KEY_ORIGINAL].shape

    ### Start timing
    tstart = time.perf_counter()

    ### Close file so subprocesses can open (r) it safely
    gv.f.close()

    ### Create output array
    c_video_out = RawArray(ctypes.c_uint8, t*x*y*c)
    video_out = np.frombuffer(c_video_out, dtype=np.uint8).reshape((t,x,y,c))
    video_out[:,:,:] = 0

    ### Progress list
    manager = Manager()
    progress = manager.list()

    seglen = 10
    segs = np.arange(0, t, seglen, dtype=int)
    print('Video frames', t)
    print('Video segments', segs)
    gv.statusbar.startProgress('Median subtraction + Range Normalization...', len(segs) * 2 - 1)

    process_num = cpu_count()-2
    print('Using {} subprocesses'.format(process_num))
    with Pool(process_num, initializer=init_worker, initargs=(c_video_out, (t,x,y,c), progress, seglen, gv.filepath)) as p:

        print('Calculate medians')
        r1 = p.map_async(worker_calc_pixel_median, segs)
        while not(r1.ready()):
            time.sleep(1/10)
            gv.statusbar.setProgress(len(progress))


    gv.statusbar.endProgress()

    gv.f = h5py.File(gv.filepath, 'a')

    print('Time for execution:', time.perf_counter()-tstart)

    ### Save to file
    gv.statusbar.startBlocking('Saving...')
    gv.f[gv.KEY_PROCESSED][:] = video_out
    gv.statusbar.setReady()

################
## Worker functions

# Globals
c_video_out = None
c_minmax_val = None
data_shape = None
video_out = None
finished_idcs = None
filepath = None
f = None

def init_worker(arr2, dshape, idx_list, chunk, fpath):
    global c_video_out, video_out, data_shape, finished_idcs, chunksize, filepath, f
    chunksize = chunk
    filepath = fpath
    f = h5py.File(filepath, 'r')
    c_video_out = arr2
    data_shape = dshape
    video_out = np.frombuffer(c_video_out, dtype=np.uint8).reshape(data_shape)
    finished_idcs = idx_list

def worker_calc_pixel_median(start_idx):
    global video_out, data_shape, finished_idcs, chunksize, f
    end_idx = start_idx + chunksize
    print('Slice {} to {}'.format(start_idx, end_idx))

    finished_idcs.append(('median_started', start_idx, end_idx))


    medrange = 20
    for i in range(start_idx, end_idx):

        start = i - medrange
        end = i + medrange
        if start < 0:
            end -= start
            start = 0

        if end > data_shape[0]:
            start -= end - data_shape[0]
            end = data_shape[0]

        video_out[i,:,:,:] = median_norm(f['original'][i, :, :, :], f['original'][start:end, :, :, :])

    print('Slice {} to {} finished'.format(start_idx, end_idx, 'finished'))

    finished_idcs.append(('median_finished', start_idx, end_idx))
    return start_idx

def median_norm(frame, slice):
    out = frame - np.median(slice, axis=0).astype(np.float32)
    out -= out.min()
    out /= out.max()
    return (out * (2**8-1)).astype(np.uint8)