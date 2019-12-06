from math import sin

from numba import njit, prange, set_num_threads

# _launch_threads()

# Set this to something less than your number of cores
N_THREADS = 4

@njit(parallel=True)
def foo(n):
    acc1 = 0
    acc2 = 0
    print("region 1 start, threads in use is default")
    for i in prange(n):
        set_num_threads(2) # should have no effect/be an error?
        acc1 += sin(i)

    # region 1 end
    set_num_threads(N_THREADS)

    print("region 2, threads in use is", N_THREADS)
    for i in prange(n):
        acc2 += sin(i)
    # region 2 end
    return acc1, acc2

N = 1000000000

print(foo(N))
