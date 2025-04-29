# 零基础大模型推理-配套代码

> 请注意阅读每个子文件下的 README.md

## 环境配置

课程提供的服务器已经配置好了环境，如果你想从头开始，以下是一些配环境的注意事项：

1. 确保 pytorch 的 CUDA 版本的 CUDA-Toolkit 的版本一致（谨慎使用最新版的 CUDA-Toolkit，因为其可能没有正式的 pytorch 版本与之对应，建议先安装 pytorch，再根据 pytorch 的 CUDA 版本安装 CUDA-Toolkit）
2. 除了 pytorch，你还需要安装 `pip install ninja numpy pandas`
3. 如果你使用 conda，可能会遇到 conda 内某些 .so 版本不够新，导致运行时报错，你可以通过 `conda install -c conda-forge libstdcxx-ng` 来解决这个问题

> 服务器使用了最新的 CUDA-Toolkit，并且使用 nightly 版本的 pytorch，如果你想使用最新的 CUDA-Toolkit，你也可以像我这样做

## 第一节：面向大模型推理的大模型，服务器与编程语言基础

代码：

- [`native-ctypes`](./native-ctypes/)

## 第二节：大模型基础与并行计算初探

代码：

- [`native-ctypes`](./native-ctypes/)
- [`pytorch-extension`](./pytorch-extension/)