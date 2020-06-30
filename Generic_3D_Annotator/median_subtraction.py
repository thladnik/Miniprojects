import ctypes
from multiprocessing import cpu_count, RawArray, Pool, Manager
import numpy as np
import time

import gvars

def run(video, report_handle=None, app=None):
    print('Run Median subtraction')

    t,x,y,c = video.shape

    tstart = time.perf_counter()
    c_video_in = RawArray(ctypes.c_int16, t*x*y*c)
    video_in = np.frombuffer(c_video_in, dtype=np.int16).reshape((t,x,y,c))
    video_in[:,:,:,:] = video

    c_video_out = RawArray(ctypes.c_int16, t*x*y*c)
    video_out = np.frombuffer(c_video_out, dtype=np.int16).reshape((t,x,y,c))
    video_out[:,:,:] = 0

    ### Progress list
    manager = Manager()
    progress = manager.list()

    report_handle.setMinimum(0)

    seglen = 40
    segs = np.arange(0, t, seglen, dtype=int)
    print('Video frames', t)
    print('Video segments', segs)
    report_handle.setMaximum(len(segs) * 4 - 1)

    process_num = cpu_count()-2
    print('Using {} subprocesses'.format(process_num))
    with Pool(process_num, initializer=init_worker, initargs=(c_video_in, c_video_out, (t,x,y,c), progress, seglen)) as p:

        print('Calculate medians')
        r1 = p.map_async(worker_calc_pixel_median, segs)
        while not(r1.ready()):
            report_handle.setValue(len(progress))
            app.processEvents()

        #video_out -= video_out.min(axis=(1, 2, 3))[:,np.newaxis, np.newaxis, np.newaxis]
        #video_out = (video_out / video_out.max(axis=(1, 2, 3))[:,np.newaxis, np.newaxis, np.newaxis] * 255).astype(np.int16)

        print('Normalize to range')
        r2 = p.map_async(worker_norm_frame, segs)
        report_handle.setValue(0)
        while not(r2.ready()):
            report_handle.setValue(len(progress))
            app.processEvents()


    print('Time for execution:', time.perf_counter()-tstart)
    return video_out.astype(np.uint8)

################
## Worker functions

# Globals
c_video_in = None
c_video_out = None
c_minmax_val = None
data_shape = None
video_in = None
video_out = None
finished_idcs = None

def init_worker(arr1, arr2, dshape, idx_list, chunk):
    global c_video_in, c_video_out, video_in, video_out, data_shape, finished_idcs, chunksize
    chunksize = chunk
    c_video_in = arr1
    c_video_out = arr2
    data_shape = dshape
    video_in = np.frombuffer(c_video_in, dtype=np.int16).reshape(data_shape)
    video_out = np.frombuffer(c_video_out, dtype=np.int16).reshape(data_shape)
    finished_idcs = idx_list

def worker_calc_pixel_median(start_idx):
    global c_video_in, video_in, video_out, data_shape, finished_idcs, chunksize
    end_idx = start_idx + chunksize
    print('Slice {} to {}'.format(start_idx, end_idx))

    finished_idcs.append(('median_started', start_idx, end_idx))

    medrange = 100
    for i in range(start_idx, end_idx):

        start = i - medrange
        end = i + medrange
        if start < 0:
            end -= start
            start = 0

        if end > data_shape[0]:
            start -= end - data_shape[0]
            end = data_shape[0]


        video_out[i, :, :, :] = video_in[i, :, :, :] - np.median(video_in[start:end, :, :, :], axis=0).astype(np.int16)

    print('Slice {} to {} finished'.format(start_idx, end_idx, 'finished'))

    finished_idcs.append(('median_finished', start_idx, end_idx))
    return start_idx

def worker_norm_frame(start_idx):
    global c_video_in, video_in, video_out, minmax_val, data_shape, finished_idcs, chunksize
    end_idx = start_idx + chunksize
    print('Slice {} to {}'.format(start_idx, end_idx))
    finished_idcs.append(('norm_started', start_idx, end_idx))


    video_out[start_idx:end_idx, :, :, :] -= video_out[start_idx:end_idx, :, :, :].min(axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    video_out[start_idx:end_idx, :, :, :] = (video_out[start_idx:end_idx, :, :, :] / video_out[start_idx:end_idx, :, :, :].max(axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis] * 255).astype(np.int16)


    finished_idcs.append(('norm_finished', start_idx, end_idx))

    print('Slice {} to {} finished'.format(start_idx, end_idx))