import torch
import triton
import triton.language as tl
import time
from typing import Optional

# 4096 * 4096
# 1024

from torch.utils.cpp_extension import load
torch.set_grad_enabled(False)

import os

# os.environ["TORCH_CUDA_ARCH_LIST"] = "sm_89"

lib = load(
    name="sgemv_lib",
    sources=[
        "gemv.cu",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

@triton.jit
def gemv_kernel(a_ptr, b_ptr, out_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.full((BLOCK_SIZE_N, ), value=pid, dtype=tl.int32)
    col = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = col < N
    accum = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k in range(K):
        a = tl.load(a_ptr + row * K + k, mask=mask)
        b = tl.load(b_ptr + k * N + col, mask=mask)
        accum += a * b
    tl.store(out_ptr + row * N + col, accum, mask=mask)
    

def gemv(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    if out is None:
        out = torch.empty((M, N), device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    a_ptr = a.contiguous().view(-1)
    b_ptr = b.contiguous().view(-1)
    gemv_kernel[grid](a_ptr, b_ptr, out, M, N, K, BLOCK_SIZE_M=1, BLOCK_SIZE_N=128)
    return out

def torch_gemv(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    if out is None:
        out = torch.empty((M, N), device=a.device)
    out.copy_(torch.matmul(a, b))
    # out = torch.matmul(a, b)
    return out

def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 2,
    iters: int = 20,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        if out is not None:
            perf_func(a, b, out)
        else:
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    # TFLOPS = (2 * M * N * K) * 1e-9 / (mean_time)
    return mean_time

# example
M, N, K = 1, 8192, 8192
x = torch.randn((M, K), device='cuda')
A = torch.randn((K, N), device='cuda')
out_triton = torch.empty((M, N), device='cuda')
out_cuda = torch.empty((M, N), device='cuda')
out_cuda_smem = torch.empty((M, N), device='cuda')
out_torch = torch.empty((M, N), device='cuda')

WARMUP = 10
ITERS = 100

time_triton = run_benchmark(
    gemv,
    x,
    A,
    "GEMV Triton",
    out=out_triton,
    warmup=WARMUP,
    iters=ITERS
)

time_cuda = run_benchmark(
    lib.sgemv,
    x,
    A,
    "GEMV CUDA",
    out=out_cuda,
    warmup=WARMUP,
    iters=ITERS
)

time_cuda_smem = run_benchmark(
    lib.sgemv_smem,
    x,
    A,
    "GEMV CUDA SMEM",
    out=out_cuda_smem,
    warmup=WARMUP,
    iters=ITERS
)

time_torch = run_benchmark(
    torch_gemv,
    x,
    A,
    "GEMV PyTorch",
    out=out_torch,
    warmup=WARMUP,
    iters=ITERS
)

print(f"GEMV Triton: {time_triton:.5f} ms")
print(f"GEMV CUDA: {time_cuda:.5f} ms")
print(f"GEMV CUDA SMEM: {time_cuda_smem:.5f} ms")
print(f"GEMV PyTorch: {time_torch:.5f} ms")

# print(f"Max error (Triton): {(out_triton - out_torch).abs().mean().item():.6f}")
# print(f"Max error (CUDA): {(out_cuda - out_torch).abs().mean().item():.6f}")
# print(f"Max error (CUDA SMEM): {(out_cuda_smem - out_torch).abs().mean().item():.6f}")

# import numpy as np

# np.savetxt("out_cuda_smem.txt", out_cuda_smem.cpu().numpy(), fmt="%.6f")
# np.savetxt("out_torch.txt", out_torch.cpu().numpy(), fmt="%.6f")

# 128 SM
# 4096 / 128 = 32
# 64 SM