import os
import signal
import numba
import math

from numba.npyufunc.parallel import set_num_threads

N_THREADS = 8

@numba.njit(parallel=True)
def test(n):
    a = 0
    for i in numba.prange(n):
        a += math.sin(i)
    return a

def worker(i):
    print('Entering worker(%s)' % (i))

    N = 1000000000

    res = test(N)
    print("Worker(%s) computed %s" % (i, res))

    print('Leaving worker(%s)' % (i))

def main(N = 8):

    # WISH: set_num_threads(1) so that inc would be single threaded
    # both in the main and child processes

    set_num_threads(1)

    # fire up N - 1 workers in child processes
    child_pids = []
    for i in range(1, N):
        pid = os.fork()
        if pid == 0:
            worker(i)
            os._exit(0)
        else:
            child_pids.append(pid)
    # fire up a worker in main process:
    worker(0)

    # wait for child processes to finish:
    while child_pids:
        p = child_pids.pop()
        print('waiting for child process (pid=%s)' % (p))
        pid, exit_status = os.waitpid(p, 0)
        if exit_status:
            print('child process (pid=%s) exited with status=%s' % (pid, exit_status))
            while child_pids:  # kill all other childs
                p = child_pids.pop()
                print('killing child process (pid=%s)' % p)
                os.kill(p, signal.SIGKILL)

    # set_num_threads(8) so that test would use 8 threads
    set_num_threads(N_THREADS)

    worker(0)

if __name__ == '__main__':
    main()
