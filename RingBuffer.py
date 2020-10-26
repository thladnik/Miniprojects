import numpy as np
import multiprocessing as mp
import ctypes
import time

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
    def __init__(self, size, dtype, *args):
        BufferAttribute.__init__(self, *args)

        assert isinstance(size, tuple), 'size has to be tuple'
        self._size = size

        self._dtype = dtype

        init = int(np.product((self._length,) + self._size))
        self._raw = mp.Array(self._dtype[0], init)
        self._data: np.ndarray = None

    def build(self):
        ar = np.frombuffer(self._raw.get_obj(), self._dtype[1])
        self._data = ar.reshape((self._length,) + self._size)

    def _read(self, start_idx, internal_idx):
        if start_idx >= 0:
            return self._data[start_idx:internal_idx].copy()
        else:
            ar1 = self._data[start_idx:]
            ar2 = self._data[:internal_idx]
            return np.concatenate((ar1, ar2))

    def read(self, last=1, use_lock=True):
        assert last < self._length, 'Trying to read more values than storedin buffer'

        internal_idx = self._index % self._length

        if self._index <= last:
            return [-1], None

        start_idx = internal_idx - last

        # Read without lock
        if not(use_lock):
            return self._get_index_list(last), self._read(start_idx, internal_idx)

        # Read with lock
        with self._raw.get_lock():
            return self._get_index_list(last), self._read(start_idx, internal_idx)


    def write(self, value):
        with self._raw.get_lock():
            self._data[self._index % self._length] = value

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
        #print('Worker write time: ', '{:.10f}'.format(time.perf_counter()-t))
        b.next()
        i += 1



if __name__ == '__main__':

    manager = mp.Manager()

    print('Init')
    b = RingBuffer()
    print('Make buffers')
    b.o1 = ObjectAttribute()
    b.a1 = ArrayAttribute(ar_size, dtype=(ctypes.c_uint8, np.uint8))

    p = mp.Process(target=worker, args=(b,))
    # FORK!
    print('Fork')
    p.start()

    print('Build')
    b.build()

    print('Run')
    while True:
        t = time.perf_counter()
        idcs, ar = b.a1.read(50, use_lock=True)
        _, randints = b.o1.read(20)
        t = time.perf_counter()-t

        if ar is None:
            print('Is None')
            continue

        print('Main read time:', '{:.5f}'.format(t), f'{ar.sum()}', f'Randint {randints}')

    p.join()
