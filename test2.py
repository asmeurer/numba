import numpy as np
from numba import njit, prange

@njit(parallel=True)
def two_d_array_reduction_prod_parallel(n):
    shp = (10, 10)
    result1 = np.ones(shp, np.float_)
    tmp_shp = (n, 10, 10)
    tmp = np.ones(tmp_shp, np.float_)

    for i in range(n):
        tmp[i] = i + 1

    for i in prange(n):
        result1 += tmp[i]

    return result1

def two_d_array_reduction_prod_sequential(n):
    shp = (10, 10)
    result1 = np.ones(shp, np.float_)
    tmp_shp = (n, 10, 10)
    tmp = np.ones(tmp_shp, np.float_)

    for i in range(n):
        tmp[i] = i+1

    for i in range(n):
        result1 /= tmp[i]

    return result1

print(two_d_array_reduction_prod_sequential(10))
print(njit(two_d_array_reduction_prod_sequential)(10))
print(two_d_array_reduction_prod_parallel(10))
