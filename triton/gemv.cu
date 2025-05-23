#include <torch/extension.h>
#include <torch/types.h>

template <int BLOCK_SIZE>
__global__ void sgemv_kernel(float* a, float* b, float* out, int M, int N, int K) {
    int row = blockIdx.x;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    if (col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            float a_val = a[row * K + k];
            float b_val = b[k * N + col];
            sum += a_val * b_val;
        }
        out[row * N + col] = sum;
    }
}

template <int BLOCK_SIZE, int SMEM_SIZE>
__global__ void sgemv_smem_kernel(float* a, float* b, float* out, int M, int N, int K) {
    int row = blockIdx.x;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    __shared__ float a_smem[SMEM_SIZE];
    float sum = 0.0;
    for (int k_iter = 0; k_iter < K; k_iter += SMEM_SIZE) {
        for (int k = 0; k < SMEM_SIZE; k += BLOCK_SIZE) {
            int smem_idx = threadIdx.x + k;
            int col_idx = k_iter + smem_idx;
            if (col_idx < K) {
                a_smem[smem_idx] = a[row * K + col_idx];
            }
        }
        __syncthreads();
        if (col < N) {
            for (int k = 0; k < SMEM_SIZE; k++) {
                float a_val = a_smem[k];
                // float a_val = a[k_iter + k];
                float b_val = b[(k_iter + k) * N + col];
                sum += a_val * b_val;
            }
        }
    }
    if (col < N) {
        out[row * N + col] = sum;
    }
}

#define BLOCK_SIZE 128

void sgemv(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    int M = a.size(0);
    int N = b.size(1);
    int K = a.size(1);
    dim3 block(BLOCK_SIZE);
    dim3 grid(M, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sgemv_kernel<BLOCK_SIZE><<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
}

void sgemv_smem(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    int M = a.size(0);
    int N = b.size(1);
    int K = a.size(1);
    dim3 block(BLOCK_SIZE);
    dim3 grid(M, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sgemv_smem_kernel<BLOCK_SIZE, 1024><<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemv", &sgemv, "SGEMV kernel");
    m.def("sgemv_smem", &sgemv_smem, "SGEMV kernel with shared memory");
}