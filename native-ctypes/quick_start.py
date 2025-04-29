import ctypes
import numpy as np
import os

# 加载当前目录下的动态链接库 libsquare.so
lib = ctypes.cdll.LoadLibrary(os.path.abspath("libsquare.so"))

# 指定 C 函数的参数类型
square_array = lib.square_array
square_array.argtypes = [ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float),
                         ctypes.c_int]

# 创建输入和输出数组
x = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.zeros_like(x)

# 调用 C 函数进行平方计算
square_array(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
             y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
             x.size)

print("Input:", x)
print("Squared:", y)
