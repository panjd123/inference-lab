#include <chrono>
#include <iostream>
#include "square.h"

double benchmark(void (*func)(const float*, float*, int), const float* input, float* output, int size, int iter) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; ++i) {
        func(input, output, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return ms / iter;
}

double cuda_benchmark(void (*func)(const float*, float*, int), const float* input, float* output, int size, int iter);

int main() {
    int N = 1048576;
    float* input = new float[N];
    float* output = new float[N];
    for (int i = 0; i < N; ++i) {
        input[i] = static_cast<float>(i);
    }

    const int iter = 1000;
    double ms1 = benchmark(square_array, input, output, N, iter);
    double ms2 = benchmark(parallel_square_array, input, output, N, iter);
    double ms3 = cuda_benchmark(native_cuda_square_array, input, output, N, iter);
    printf("%20s: %f ms\n", "CPU", ms1);
    printf("%20s: %f ms\n", "CPU(parallel)", ms2);
    printf("%20s: %f ms\n", "GPU(cuda)", ms3);

    delete[] input;
    delete[] output;
}