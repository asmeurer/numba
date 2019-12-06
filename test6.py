from numba import njit, set_num_threads, get_num_threads

import numpy as np

# Set this to something less than your number of cores
N_THREADS = 4

@njit(parallel=True)
def foo(n):
    print("region 1 start, threads in use is default")
    a = np.arange(n)
    b = np.sin(np.sin(np.sin(np.sin(np.sin(np.sin(np.sin(a) + 1))))))
    # region 1 end

    set_num_threads(N_THREADS)

    print("region 2, threads in use is", get_num_threads())
    c = np.sin(np.sin(np.sin(np.sin(np.sin(a) + 2))))
    # region 2 end
    return b, c

N = 1000000000

foo(N)
