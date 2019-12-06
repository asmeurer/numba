from numba import njit, prange
from numba.npyufunc.parallel import get_thread_count, set_num_threads

import numpy as np

import math

@njit
def value(i):
    return i**2 - i//2 + math.sin(i)

@njit(parallel=True)
def test(n):
    a = 0
    for i in prange(n):
        a += math.sin(i)
    return a

N = 1000000000
# N = 10
NUM_THREADS = get_thread_count()
print(f"Running with all threads ({NUM_THREADS})")
# CFUNCTYPE(None, c_int)(tbbpool.set_num_threads)(16)
print(test(n=N))

print("Running with 4 threads")
set_num_threads(4)
print(test(n=N))

# print("serial result")
# def serial():
#     return np.sum(np.sin(np.arange(N)))
# print(serial())
