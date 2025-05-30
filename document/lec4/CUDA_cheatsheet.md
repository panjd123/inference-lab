# CUDA cheatsheet

## CUDA 中的常见 API

### host 端内存相关

- `cudaMalloc`: 在设备上分配内存
- `cudaFree`: 释放设备内存
- `cudaMemcpy`: 在主机和设备之间复制内存

### Block 粒度（被调度到某个 SM 上）

- `__shared__`: 声明共享内存变量
- `__syncthreads()`: 同步线程，确保所有线程都到达此点

### Warp 粒度

- `__shfl_sync`: 在同一个 warp 内的线程之间交换数据
- 还有很多类似的指令，比如 `__shfl_down_sync` 等等，建议用 AI 来学习

### device 端的访存相关

- 用 `float4` 来访问，以此触发 `ld.global.v4.f32` 而不是 `ld.global.f32` 指令

## 优化方法

### 访存相关

- 访存对齐：确保数据在内存中的对齐方式与访问方式一致
- 用更大的粒度来访问内存，比如用 `float4` 来访问 `float` 数组
- 避免不必要的访存，比如在寄存器/共享内存中缓存数据，避免重复访存（通常还涉及算法的改动）
- 避免 bank conflict：在共享内存中，避免多个线程访问同一个 bank 的数据

### 控制流相关

#### 通信方式总结

- 跨 block 的线程得通过 global memory 通信，以及 global_memory 里显式的锁来同步
- 同一个 block 内跨 warp 的线程可以通过 `__shared__` 共享内存来通信，以及 `__syncthreads()` 来同步
- 同一个 warp 内的线程可以通过 `__shfl_sync` 来通信，通常情况下，他们已经是同步的
- 越下面的内存介质访问速度越快，同步开销越小

#### 减少分歧

同一个 Warp 内的线程尽量做相同的事，避免分支成串行的

#### 减少同步

通过设计算法，平衡并行度和同步开销

比如跨 block 的并行求和，并行度很高，但是最后同步求和的开销很大；
在 block 内求和，可以最后用 shared memory 来规约，但是可能会导致任务太小，block 的数量太少，用不满 GPU 的计算资源；
更进一步，可以在 warp 内求和，但是更可能导致 GPU 资源的浪费

通常情况下，我们要在尽量打满 GPU 资源的前提下，减少同步开销，比如即使我们需要跨 block 同步了，但是如果不这样，没办法用满 SM，那大概也是值得的。 

### 调参以及多多尝试

- 通过 profiler 来分析性能瓶颈（Nsight Compute）
- 不同的问题规模往往需要不同的算法设计
- 多调一调 BLOCK_SIZE 等参数，往往有奇效（）
- 每种方法都可以尝试，理论和现实可能不相符（由于 GPU 不是一个简单的串行设备，其还涉及我们不可控的调度因素，进而导致不同硬件单元在执行的时候以不同的方式互相掩盖，总之就是即使是改变了一行汇编代码，其硬件执行可能都有巨大的不同，所以多多尝试）

## 助教最常翻的文档们

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [arithmetic-instructions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions)
- [Half Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF.html)
- [Single Precision Mathematical Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html)
- [Single Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
- [Parallel Thread Execution ISA Version 8.8](https://docs.nvidia.com/cuda/parallel-thread-execution/)，这个本节课大概率用不上，因为他是硬件高度相关的

## 编写 CUDA 的一些提醒或技巧

- 永远记得检查 CUDA 错误，比如 `cudaGetLastError()`，CUDA 报错了不会告诉你，除非你主动问，不要你写了一个比 baseline 快 100x 的代码，最后发现是报错了直接崩溃了
- 同上，永远记得做结果的数值检查
- cuda 有很多好工具，比如 Nsight Compute 和 Nsight Systems 可以帮助你分析瓶颈，`cuda-gdb` 可以帮助你调试 CUDA 程序，[`memcheck`](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#using-memcheck) 可以帮助你检查内存泄漏或者是越界访问
- 如果你会 Triton，用 Triton 验证一些想法是个不错的选择，但是 Triton 在细粒度的控制上可能不如 CUDA 灵活