"""
https://github.com/thladnik/Miniprojects/blob/master/RingBuffer.py - Basic implementation of a fast circular buffer..
Copyright (C) 2020 Tim Hladnik
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import multiprocessing as mp
import ctypes
import time
from typing import List

class DummyLockContext:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class BufferAttribute:

    def __init__(self, length=100):
        self._length = length
        self._data = None
        self._index = None

    def build(self):
        pass

    def get(self, current_idx):
        self._index = current_idx
        return self

    def read(self, last=-1):
        raise NotImplementedError('')

    def write(self, value):
        raise NotImplementedError('')

    def _get_index_list(self, last):
        return list(range(self._index - last, self._index))


class ArrayAttribute(BufferAttribute):
    def __init__(self, size, dtype, chunked=False, chunk_size=None, *args):
        BufferAttribute.__init__(self, *args)

        assert isinstance(size, tuple), 'size has to be tuple'
        self._size = size
        self._dtype = dtype
        self._chunked = chunked
        self._chunk_size = chunk_size

        if self._chunked and self._chunk_size is not None:
            assert self._length % self._chunk_size == 0, 'Chunk size of buffer does not match its length'

        # Automatically determine chunk_size
        if self._chunked and self._length < 10:
            self._chunked = False
            print('WARNING', 'Automatic chunking disabled (auto)', 'Buffer length too small.')

        if self._chunked and self._chunk_size is None:
            for s in range(self._length // 10, self._length):
                if self._length % s == 0:
                    self._chunk_size = s
                    break

            if self._chunk_size is None:
                self._chunk_size = self._length // 10
                self._length = 10 * self._chunk_size
                print('WARNING', 'Unable to find suitable chunk size.', 'Resize buffer to match closest length.')

        # This should be int
        self._chunk_num = None
        if self._chunked:
            self._chunk_num = self._length // self._chunk_size

        # Init data structures
        if self._chunked:
            init = int(np.product((self._chunk_size,) + self._size))
            self._raw: List[mp.Array] = list()
            for i in range(self._chunk_num):
                self._raw.append(mp.Array(self._dtype[0], init))
            self._data: List[np.ndarray] = list()
        else:
            init = int(np.product((self._length,) + self._size))
            self._raw: mp.Array = mp.Array(self._dtype[0], init)
            self._data: np.ndarray = None

    def _build_array(self, raw, length):
        np_array = np.frombuffer(raw.get_obj(), self._dtype[1])
        return np_array.reshape((length,) + self._size)

    def _get_lock(self, chunk_idx, use_lock):
        if not(use_lock):
            return DummyLockContext()

        if chunk_idx is None:
            lock = self._raw.get_lock
        else:
            lock = self._raw[chunk_idx].get_lock

        return lock()

    def build(self):
        if self._chunked:
            for raw in self._raw:
                self._data.append(self._build_array(raw, self._chunk_size))
        else:
            self._data = self._build_array(self._raw, self._length)

    def _read(self, start_idx, end_idx, use_lock):
        if self._chunked:
            start_chunk = start_idx // self._chunk_size
            chunk_start_idx = start_idx % self._chunk_size
            end_chunk = end_idx // self._chunk_size
            chunk_end_idx = end_idx % self._chunk_size

            # Read within one chunk
            if start_chunk == end_chunk:
                with self._get_lock(start_chunk, use_lock):
                    return self._data[start_chunk][chunk_start_idx:chunk_end_idx]

            # Read across multiple chunks
            np_arrays = list()
            with self._get_lock(start_chunk, use_lock):
                np_arrays.append(self._data[start_chunk][chunk_start_idx:])
            for ci in range(start_chunk+1, end_chunk):
                with self._get_lock(ci, use_lock):
                    np_arrays.append(self._data[ci][:])
            with self._get_lock(end_chunk, use_lock):
                np_arrays.append(self._data[end_chunk][:chunk_end_idx])

            return np.concatenate(np_arrays)

        else:
            with self._get_lock(None, use_lock):
                if start_idx >= 0:
                    return self._data[start_idx:end_idx].copy()
                else:
                    ar1 = self._data[start_idx:]
                    ar2 = self._data[:end_idx]
                    return np.concatenate((ar1, ar2))

    def read(self, last=1, use_lock=True):
        assert last < self._length, 'Trying to read more values than stored in buffer'

        internal_idx = self._index % self._length

        if self._index <= last:
            return [-1], None

        start_idx = internal_idx - last

        # Read without lock
        #if not(use_lock):
        return self._get_index_list(last), self._read(start_idx, internal_idx, use_lock)

        # Read with lock
        #with self._raw.get_lock():
        #return self._get_index_list(last), self._read(start_idx, internal_idx, use_lock)


    def write(self, value):
        internal_idx = self._index % self._length
        if self._chunked:
            chunk_idx = internal_idx // self._chunk_size
            idx = internal_idx % self._chunk_size

            with self._get_lock(chunk_idx, True):
                self._data[chunk_idx][idx] = value
        else:
            with self._get_lock(None, True):
                self._data[internal_idx] = value

    def __setitem__(self, key, value):
        self._data[key % self._length] = value


class ObjectAttribute(BufferAttribute):
    def __init__(self, *args):
        BufferAttribute.__init__(self, *args)
        global manager

        self._data = manager.list([None] * self._length)

    def read(self, last=1) -> (list, np.ndarray):
        internal_idx = self._index % self._length

        start_idx = internal_idx - last
        if start_idx >= 0:
            return self._get_index_list(last), self._data[start_idx:internal_idx]
        else:
            return self._get_index_list(last), self._data[start_idx:] + self._data[:internal_idx]

    def write(self, value):
        self._data[self._index % self._length] = value

    def __setitem__(self, key, value):
        self._data[key % self._length] = value


class RingBuffer:

    def __init__(self):
        self.__dict__['current_idx'] = mp.Value(ctypes.c_uint64)

    def build(self):
        for attr_name, obj in self.__dict__.items():
            if not (attr_name.startswith('_attr_')):
                continue

            obj.build()

    def set_index(self, new_idx):
        self.__dict__['current_idx'].value = new_idx

    def get_index(self):
        return self.__dict__['current_idx'].value

    def __setattr__(self, key, value):
        assert issubclass(value.__class__, BufferAttribute)
        self.__dict__[f'_attr_{key}'] = value

    def __getattr__(self, item) -> BufferAttribute:
        # Return
        try:
            return self.__dict__[f'_attr_{item}'].get(self.get_index())
        except:
            # Fallback for serialization
            self.__getattribute__(item)

    def next(self):
        self.set_index(self.get_index() + 1)


ar_size = (600,700,3)
def worker(b: RingBuffer):

    b.build()

    ar = np.ones(ar_size)
    i = 1
    while True:

        b.o1.write(np.random.randint(0,50))
        ar2 = i * ar
        t = time.perf_counter()
        b.a1.write(ar2)
        print('Worker WRITE time: ', '{:.10f}'.format(time.perf_counter()-t))
        b.next()
        i += 1


if __name__ == '__main__':

    manager = mp.Manager()

    print('Init')
    b = RingBuffer()
    print('Make buffers')
    b.o1 = ObjectAttribute()
    b.a1 = ArrayAttribute(ar_size, dtype=(ctypes.c_uint8, np.uint8), chunked=True)

    p = mp.Process(target=worker, args=(b,))
    # FORK!
    print('Fork')
    p.start()

    print('Build')
    b.build()

    print('Run')
    while True:
        t = time.perf_counter()
        idcs, ar = b.a1.read(90, use_lock=True)
        _, randints = b.o1.read(20)
        t = time.perf_counter()-t

        if ar is None:
            print('Is None')
            continue

        #print('Main READ time:', '{:.5f}'.format(t), f'{ar.sum()}', f'Randint {randints}')

    p.join()
