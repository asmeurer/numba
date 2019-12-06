import os
os.environ['NUMBA_NUM_THREADS'] = '4'
import numba
import numpy as np
import time


# # Select numba threading layer
# numba.config.THREADING_LAYER = 'tbb'

print("Using threading backend", numba.config.THREADING_LAYER)

@numba.guvectorize(['void(float64[:])'],
                   '(n)',
                   nopython=True,
                   target='parallel')
def inc(x):
    for i in range(10000):
        x[:] = np.sin(x)

def runme(x):
    print('Using numba threading layer: {}'.format(numba.threading_layer()))
    num_threads = len(os.listdir('/proc/{}/task'.format(os.getpid())))
    print('Num threads before inc(x) = {}'.format(num_threads))

    res = inc(x)
    num_threads = len(os.listdir('/proc/{}/task'.format(os.getpid())))
    print('Num threads after inc(x) = {}'.format(num_threads))
    return res

if __name__ == '__main__':
    K = 1000
    np.random.seed(1)
    x = np.random.random(K * K).reshape(K, K)

    print(x.sum())
    runme(x)
    print(x.sum())

    numba.set_num_threads(1)
    runme(x)
    print(x.sum())
