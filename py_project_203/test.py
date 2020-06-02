import time
import numpy as np
from numba import jit


@jit(nopython=True)
def insertion_sort_numpy(arr):
    for i in range(len(arr)):
        cursor = arr[i]
        pos = i

        while pos > 0 and arr[pos-1] > cursor:
            arr[pos] = arr[pos-1]
            pos = pos-1
        arr[pos] = cursor
    return arr


# 插入排序算法
def insertion_sort(arr):
    for i in range(len(arr)):
        cursor = arr[i]
        pos = i

        while pos > 0 and arr[pos-1] > cursor:
            arr[pos] = arr[pos-1]
            pos = pos-1
        arr[pos] = cursor
    return arr


if __name__ == '__main__':
    list_of_numbers = np.random.randint(1, 10000, 10000)

    t1 = time.time()
    result1 = insertion_sort(list_of_numbers)
    t2 = time.time()
    result2 = insertion_sort_numpy(list_of_numbers)
    t3 = time.time()
    run_time1 = t2 - t1
    run_time2 = t3 - t2
    print('tatol time1={}'.format(run_time1))
    print('tatol time2={}'.format(run_time2))