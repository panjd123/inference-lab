#include <omp.h>
#include <chrono>
#include <iostream>
#include "ascii/ascii.h"
#include "square.h"

double benchmark(void (*func)(const float*, float*, int), const float* input, float* output, int size, int iter) {
    for (int i = 0; i < iter / 10; ++i) {
        func(input, output, size);
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; ++i) {
        func(input, output, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return ms / iter;
}

double cuda_benchmark(void (*func)(const float*, float*, int), const float* input, float* output, int size, int iter);

using namespace ascii;

int main() {
    int N = 1048576;
    float* input = new float[N];
    float* output = new float[N];
    for (int i = 0; i < N; ++i) {
        input[i] = static_cast<float>(i);
    }

    const int iter = 10000;
    double ms1 = benchmark(square_array, input, output, N, iter);
    int max_threads = omp_get_max_threads();
    double parallel_ms[max_threads];
    for (int i = 0; i < max_threads; ++i) {
        omp_set_num_threads(i + 1);
        parallel_ms[i] = benchmark(parallel_square_array, input, output, N, iter);
    }
    double ms3 = cuda_benchmark(native_cuda_square_array, input, output, N, iter);
    std::cout << std::format("{:8} {:10.5f}", "CPU", ms1) << std::endl;
    for (int i = 0; i < max_threads; ++i) {
        std::cout << std::format("{:8} {:10.5f}", "OMP " + std::to_string(i + 1), parallel_ms[i]) << std::endl;
    }
    std::cout << std::format("{:8} {:10.5f}", "CUDA", ms3) << std::endl;
    delete[] input;
    delete[] output;
    std::vector<double> data{parallel_ms, parallel_ms + max_threads};
    for (int i = 0; i < max_threads; ++i) {
        data[i] = data[i] * 1000;
    }
    Asciichart chart(data);
    std::cout << chart.height(20).Plot() << std::endl;
}