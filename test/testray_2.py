import ray

@ray.remote
def initialize():
    global nlist
    nlist = [1,2,3]
    print("try to parallel123")