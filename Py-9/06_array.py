import numpy as np

arr = np.array([1, 5, 2, 9, 10])
print(arr.dtype)
print(arr)

arr = np.array([1, 5, 2, 9, 10], dtype=np.int8)
nd_arr = np.array([
    [12, 45, 78],
    [34, 56, 13],
    [12, 98, 76]
], dtype=np.int16)

print(arr.ndim)
print(nd_arr.ndim)

arr, step = np.linspace(-6, 21, num=60, retstep=True)
print(round(step, 2))


arr, step = np.linspace(-6, 21, endpoint=False, num=60, retstep=True)
print(round(step, 2))