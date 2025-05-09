#include <omp.h>
#include <torch/extension.h>
#include "fast_cuda/matmul/matmul.h"

FAST_MATMUL fast_matmul_method = FAST_MATMUL::NATIVE;

void set_fast_matmul_method(FAST_MATMUL method) {
    fast_matmul_method = method;
}

torch::Tensor square_cpu(torch::Tensor x);
torch::Tensor square_cuda(torch::Tensor x);

template <typename scalar_t>
void square_cpu_impl(scalar_t* input, scalar_t* output, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] * input[i];
    }
}

torch::Tensor square_cpu(torch::Tensor x) {
    auto output = torch::empty_like(x);
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "square_cpu", ([&] {
                              auto input_ptr = x.data_ptr<scalar_t>();
                              auto output_ptr = output.data_ptr<scalar_t>();
                              int size = x.numel();
                              square_cpu_impl(input_ptr, output_ptr, size);
                          }));
    return output;
}

// 接收一个 Tensor 并返回其平方结果
torch::Tensor square(torch::Tensor input) {
    if (input.device().is_cuda()) {
        return square_cuda(input);
    } else {
        return square_cpu(input);
    }
}

torch::Tensor matmul_rc_cpu(torch::Tensor a, torch::Tensor b);
torch::Tensor matmul_rc_cuda(torch::Tensor a, torch::Tensor b);

template <typename scalar_t>
void matmul_rc_cpu_impl(scalar_t* a_ptr, scalar_t* b_ptr, scalar_t* d_ptr, int m, int n, int k) {
#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            d_ptr[i * n + j] = 0;
            for (int l = 0; l < k; ++l) {
                d_ptr[i * n + j] += a_ptr[i * k + l] * b_ptr[j * k + l];
            }
        }
    }
}

torch::Tensor matmul_rc_cpu(torch::Tensor a, torch::Tensor b) {
    auto d = torch::empty({a.size(0), b.size(0)}, a.options());
    AT_DISPATCH_ALL_TYPES(a.scalar_type(), "matmul_rc_cpu", ([&] {
                              auto a_ptr = a.data_ptr<scalar_t>();
                              auto b_ptr = b.data_ptr<scalar_t>();
                              auto d_ptr = d.data_ptr<scalar_t>();
                              int m = a.size(0);
                              int n = b.size(0);
                              int k = b.size(1);
                              matmul_rc_cpu_impl(a_ptr, b_ptr, d_ptr, m, n, k);
                          }));
    return d;
}

torch::Tensor matmul_rowmajor_columnmajor(torch::Tensor a, torch::Tensor b) {
    if (a.dim() != 2 || b.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D");
    }
    if (a.size(1) != b.size(1)) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }
    if (a.device() != b.device()) {
        throw std::invalid_argument("Input tensors must be on the same device");
    }
    if (a.device().is_cuda()) {
        return matmul_rc_cuda(a, b);
    } else {
        return matmul_rc_cpu(a, b);
    }
}

// 通过 PYBIND11_MODULE 绑定 Python 接口
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", &square, "Square tensor elements");
    m.def("matmul_rowmajor_columnmajor", &matmul_rowmajor_columnmajor, "Matrix multiplication with row-major and column-major tensors");
    py::enum_<FAST_MATMUL>(m, "FastMatmulMethod")
        .value("NATIVE", FAST_MATMUL::NATIVE)
        .value("SHARED_MEMORY", FAST_MATMUL::SHARED_MEMORY)
        .value("SHARED_MEMORY_NATIVE", FAST_MATMUL::SHARED_MEMORY_NATIVE)
        .value("TILE", FAST_MATMUL::TILE)
        .export_values();
    m.def("set_fast_matmul_method", &set_fast_matmul_method, "Set the fast matmul method");
}