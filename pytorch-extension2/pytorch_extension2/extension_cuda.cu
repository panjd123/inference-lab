#include <torch/extension.h>
#include "fast_cuda/matmul/matmul.cuh"
#include "fast_cuda/square/square.cuh"

extern FAST_MATMUL fast_matmul_method;

template <typename scalar_t>
__global__ void square_cuda_kernel(scalar_t* input, scalar_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
}

template <typename scalar_t>
void square_cuda_impl(scalar_t* input, scalar_t* output, int size) {
    static const int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    square_cuda_kernel<<<grid_size, block_size>>>(input, output, size);
    cudaDeviceSynchronize();
}

torch::Tensor square_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "square_cuda", ([&] {
                              auto input_ptr = x.data_ptr<scalar_t>();
                              auto output_ptr = output.data_ptr<scalar_t>();
                              int size = x.numel();
                              square_cuda_impl(input_ptr, output_ptr, size);
                          }));
    return output;
}

template <typename scalar_t>
__global__ void matmul_rc_cuda_kernel(scalar_t* a, scalar_t* b, scalar_t* d, int m, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[col * k + i];
        }
        d[row * n + col] = sum;
    }
}

template <typename scalar_t>
void matmul_rc_cuda_impl(scalar_t* a, scalar_t* b, scalar_t* d, int m, int n, int k) {
    fast_matmul(a, b, d, m, n, k, fast_matmul_method);
    // static const dim3 block_size(16, 16);
    // dim3 grid_size((m + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
    // matmul_rc_cuda_kernel<<<grid_size, block_size>>>(a, b, d, m, n, k);
    cudaDeviceSynchronize();
}

torch::Tensor matmul_rc_cuda(torch::Tensor a, torch::Tensor b) {
    auto d = torch::empty({a.size(0), b.size(0)}, a.options());
    AT_DISPATCH_ALL_TYPES_AND(torch::ScalarType::Half, a.scalar_type(), "matmul_rc_cuda", ([&] {
                                  auto a_ptr = a.data_ptr<scalar_t>();
                                  auto b_ptr = b.data_ptr<scalar_t>();
                                  auto d_ptr = d.data_ptr<scalar_t>();
                                  int m = a.size(0);
                                  int n = b.size(0);
                                  int k = b.size(1);
                                  matmul_rc_cuda_impl(a_ptr, b_ptr, d_ptr, m, n, k);
                              }));
    return d;
}