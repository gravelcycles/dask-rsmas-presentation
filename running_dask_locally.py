from dask.distributed import as_completed, progress
from dask.distributed import LocalCluster, Client
import math
import numpy as np
import sys
from timeit import timeit

Y_DIM = 5000
NUM_JOBS = 4

vectorized_sqrt = np.vectorize(math.sqrt)


def sequential_main():
    """
    Create a large 2D numpy array, do some expensive computation on every element, return the sum
    """
    print("Creating a huge array")
    two_d_array = np.random.rand(10000, Y_DIM)

    print("Applying function sequentially")
    output = vectorized_sqrt(two_d_array)
    print("Taking the sum of our huge array")

    total = sum(sum(output))
    print(total)
    return total

def distributed_main():
    """
    Create a large 2D numpy array, do some expensive computation on every element ***IN PARALLEL***,
    return the sum.
    """
    two_d_array = np.random.rand(10000, Y_DIM)

    # Split the large array into smaller arrays along the Y axis
    # Submit each smaller array as a job
    futures = []
    for i in range(NUM_JOBS):
        start = (i * Y_DIM) // NUM_JOBS
        end = ((i + 1) * Y_DIM) // NUM_JOBS
        print([start, end])
        # Sends lots of data over the network to each worker
        future = client.submit(parallel_func, two_d_array[:, start:end])
        futures.append(future)

    progress(futures)

    total = 0
    for future in as_completed(futures):
        total += future.result()

    print(total)
    return total


def parallel_func(array):
    """
    Code to run on each worker
    """
    output = vectorized_sqrt(array)
    total = sum(sum(output))
    return total


def distributed_main2():
    futures = []
    for i in range(NUM_JOBS):
        y_dim = Y_DIM // NUM_JOBS

        # Sends very little data over the network to each worker
        future = client.submit(parallel_func2, y_dim)
        futures.append(future)

    progress(futures)
    total = 0
    for future in as_completed(futures):
        total += future.result()

    print(total)
    client.close()
    return total


def parallel_func2(dim):
    two_d_array = np.random.rand(10000, dim)
    output = vectorized_sqrt(two_d_array)
    total = sum(sum(output))
    return total


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)

    client = Client(cluster)
    print("Time to run sequential code:", timeit(stmt=sequential_main, number=1))
    print("Time to run parallel code:", timeit(stmt=distributed_main, number=1))
    print("Time to run parallel code with ~0 data transfer:", timeit(stmt=distributed_main2, number=1))


    client.close()

