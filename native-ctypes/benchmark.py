import ctypes
import numpy as np
import os
import timeit

# 加载当前目录下的动态链接库 libsquare.so
lib = ctypes.cdll.LoadLibrary(os.path.abspath("libsquare_all.so"))

# 指定 C 函数的参数类型
square_array_c_ = lib.square_array
square_array_c_.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]


def square_array_numpy(x: np.ndarray, y: np.ndarray):
    y[:] = x * x


def square_array_python(x: np.ndarray, y: np.ndarray):
    for i in range(len(x)):
        y[i] = x[i] * x[i]


def square_array_c(x: np.ndarray, y: np.ndarray):
    # 调用 C 函数
    square_array_c_(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.size,
    )


# 正确性测试
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
print(f"x: {x}")

y = np.empty_like(x)
square_array_python(x, y)
print(f"y(python): {y}")

y = np.empty_like(x)
square_array_c(x, y)
print(f"y(c): {y}")

# 性能测试
print(f"{"N":<10}{"Python":<20}{"C":<20}{"C_native":<20}{"Numpy":<20}")
iter = 20

# 1e3 -> 1e8
for N in [2**i for i in range(10, 26)]:
    x = np.random.randn(N).astype(np.float32)
    y = np.empty_like(x)
    if N > 2e6:
        timeit_python = np.inf
    else:
        timeit_python = timeit.timeit(
            "square_array_python(x, y)", globals=globals(), number=iter
        )
    timeit_c = timeit.timeit("square_array_c(x, y)", globals=globals(), number=iter)
    x_data = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y_data = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    timeit_c_native = timeit.timeit(
        "square_array_c_(x_data, y_data, N)", globals=globals(), number=iter
    )
    timeit_numpy = timeit.timeit(
        "square_array_numpy(x, y)", globals=globals(), number=iter
    )
    python_ms = timeit_python * 1000 / iter
    c_ms = timeit_c * 1000 / iter
    c_native_ms = timeit_c_native * 1000 / iter
    numpy_ms = timeit_numpy * 1000 / iter
    print(
        f"{N:<10}{python_ms:<20.5g}{c_ms:<20.5g}{c_native_ms:<20.5g}{numpy_ms:<20.5g}"
    )
