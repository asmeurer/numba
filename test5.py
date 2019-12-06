from math import sin

from numba import njit, prange, set_num_threads

# Set this to something less than your number of cores
N_THREADS = 4

@njit(parallel=True)
def test_basic(n):
    print("Running basic test of calling set_num_threads in a jitted function. It should use",
          N_THREADS, "threads.")
    a = 0
    set_num_threads(N_THREADS)
    for i in prange(n):
        a += sin(i)
    return a

N = 1000000000

print(test_basic(N))
