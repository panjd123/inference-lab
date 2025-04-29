## 如何运行

```bash
cd native-ctypes
make
python quick_start.py # ctypes 的最简例子
python benchmark.py # ctypes 的性能测试
./main # C++ 调用的例子
```

## 学习目标

- 学习 `Makefile` 文件中，`.so` 文件是怎么编译出来的
- 学习 `quick_start.py` 文件中，怎么通过 `ctype` 库在 Python 中调用 C/C++ 的函数
- 通过 `benchmark.py` 了解 Python 和 C/C++ 的速度差异

--------

- 通过 `square.cpp` 和 `square.cu` 文件学习最简单的 CPU/GPU 并行计算编程方法
- 通过 `main` 文件了解串行CPU，并行CPU，GPU的执行速度