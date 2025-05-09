import torch
from torch.utils import benchmark
import pytorch_extension2
import pandas as pd
from tqdm import tqdm
import numpy as np

num_threads = 16

datas = []
for N in tqdm([384, 768, 1024, 1536, 2048, 4096, 8192]):
    a = torch.randn(N, N, device="cpu")
    b = torch.randn(N, N, device="cpu")
    t_cpu_pytorch = benchmark.Timer(
        stmt="torch.matmul(a, b.T)",
        setup="import torch; import pytorch_extension2;",
        globals={"a": a, "b": b},
        num_threads=num_threads,
        label="torch.matmul",
        description="CPU",
    )
    t_cpu_extension = benchmark.Timer(
        stmt="pytorch_extension2.matmul_rowmajor_columnmajor(a, b)",
        setup="import torch; import pytorch_extension2;",
        globals={"a": a, "b": b},
        num_threads=num_threads,
        label="pytorch_extension2.matmul_rowmajor_columnmajor",
        description="CPU",
    )

    a_cuda = torch.randn(N, N, device="cuda")
    b_cuda = torch.randn(N, N, device="cuda")
    t_cuda_pytorch = benchmark.Timer(
        stmt="torch.matmul(a_cuda, b_cuda.T)",
        setup="import torch; import pytorch_extension2;",
        globals={"a_cuda": a_cuda, "b_cuda": b_cuda},
        num_threads=num_threads,
        label="torch.matmul",
        description="CUDA",
    )
    t_cuda_extension = benchmark.Timer(
        stmt="pytorch_extension2.matmul_rowmajor_columnmajor(a_cuda, b_cuda)",
        setup="import torch; import pytorch_extension2; pytorch_extension2.set_fast_matmul_method(pytorch_extension2.FastMatmulMethod.NATIVE);",
        globals={"a_cuda": a_cuda, "b_cuda": b_cuda},
        num_threads=num_threads,
        label="pytorch_extension2.matmul_rowmajor_columnmajor",
        description="CUDA",
    )

    t_cuda_extension_smem = benchmark.Timer(
        stmt="pytorch_extension2.matmul_rowmajor_columnmajor(a_cuda, b_cuda)",
        setup="import torch; import pytorch_extension2; pytorch_extension2.set_fast_matmul_method(pytorch_extension2.FastMatmulMethod.SHARED_MEMORY);",
        globals={"a_cuda": a_cuda, "b_cuda": b_cuda},
        num_threads=num_threads,
        label="pytorch_extension2.matmul_rowmajor_columnmajor",
        description="CUDA",
    )

    mean_cpu_pytorch = t_cpu_pytorch.timeit(10).mean * 1000 if N <= 2048 else np.inf
    mean_cpu_extension = t_cpu_extension.timeit(10).mean * 1000 if N <= 1536 else np.inf
    mean_cuda_pytorch = t_cuda_pytorch.timeit(10).mean * 1000
    mean_cuda_extension = (
        t_cuda_extension.timeit(10).mean * 1000 if N <= 4096 else np.inf
    )
    mean_cuda_extension_smem = (
        t_cuda_extension_smem.timeit(10).mean * 1000 if N <= 4096 else np.inf
    )

    datas.append(
        {
            "N": N,
            "torch.matmul": mean_cpu_pytorch,
            "pytorch_extension.matmul_rowmajor_columnmajor": mean_cpu_extension,
            "torch.matmul (CUDA)": mean_cuda_pytorch,
            "pytorch_extension.matmul_rowmajor_columnmajor (CUDA)": mean_cuda_extension,
            "pytorch_extension.matmul_rowmajor_columnmajor (CUDA, Shared Memory)": mean_cuda_extension_smem,
        }
    )

df = pd.DataFrame(datas)
print(df)
