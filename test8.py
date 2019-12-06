from numba import njit, set_num_threads, prange, get_num_threads
import numpy as np
import math

N = 500000000

import ctypes
cpython = ctypes.CDLL(None)
pthread_self_ptr = cpython.pthread_self
proto = ctypes.CFUNCTYPE(ctypes.c_ssize_t,)
pthread_self = proto(pthread_self_ptr)

@njit(parallel=True)
def child_func():
    acc = 0
    for i in prange(N):
        # set_num_threads(16)
        acc += math.sin(i)
    return acc

@njit(parallel=True)
def test_func(nthreads):
    acc = 0
    buf = np.empty(20)
    set_num_threads(nthreads)
    for i in prange(20):
        print(i)
        acc += child_func()
    print("after", get_num_threads())
    print(buf) # should be all 4's, but its not!
    return acc

mask = 4 # Set to something less than your number of cores
acc = test_func(mask)
print(acc)
