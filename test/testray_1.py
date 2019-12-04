import ray

from testray_2 import initialize

ray.init(num_cpus=4, ignore_reinit_error=True)

if __name__ == '__main__':
    ray.get([initialize.remote() for _ in range(4)])
    global nlist
    print( nlist )
