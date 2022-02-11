import numpy as np

x = np.int64(10)
print(np.iinfo(x))

x = np.uint64(10)
print(np.iinfo(x))

print(np.float16(4.121))
print(np.float16(4.123))
print(np.float16(4.124))
print(np.float16(4.125))
print(np.float16(4.126))
print(np.finfo(np.float16))

print(np.iinfo(np.int64))
x = np.uint8(-456)
print(x)
print(np.iinfo(x))
print(np.finfo(np.float32))


value = str('3.4028235e+38')
print(float(value))
#
#
# print(np.float32(0.100009))
#
# arr = np.zeros((2, 4))
# print(arr)