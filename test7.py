from numba import njit, set_num_threads, prange, get_num_threads
import numpy as np

N = 50

import ctypes
cpython = ctypes.CDLL(None)
pthread_self_ptr = cpython.pthread_self
proto = ctypes.CFUNCTYPE(ctypes.c_ssize_t,)
pthread_self = proto(pthread_self_ptr)


@njit(parallel=True)
def child_func():
    print("child_func", pthread_self(), get_num_threads())
    N = 4
    acc = 0
    for i in prange(N):
        print("child_func: prange", pthread_self(), get_num_threads())
        acc += 1
    return acc

@njit(parallel=True)
def test_func(nthreads):
    acc = 0
    mt = pthread_self()
    print("before", mt, get_num_threads())
    buf = np.empty(N)
    set_num_threads(nthreads)
    for i in prange(N):
        set_num_threads(2)
        buf[i] = child_func()
        acc += get_num_threads()
        print("in prange", pthread_self(), get_num_threads())
    print("after",get_num_threads())
    print(buf) # should be all 4's, but its not!
    return acc

mask = 3 # Set to something less than your number of cores
acc = test_func(mask)
