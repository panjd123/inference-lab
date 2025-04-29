#include "square.h"
#include <cuda_runtime.h>

// C 接口实现，将输入数组每个元素平方后写入输出数组
extern "C" void square_array(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] * input[i];
    }
}

extern "C" void parallel_square_array(const float* input, float* output, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] * input[i];
    }
}