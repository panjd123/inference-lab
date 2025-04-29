#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iostream>

__global__ void square_array_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
}

extern "C" void native_cuda_square_array(const float* input, float* output, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    square_array_kernel<<<numBlocks, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void cuda_square_array(const float* input, float* output, int size) {
    float* d_input;
    float* d_output;
    size_t bytes = size * sizeof(float);
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);
    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    native_cuda_square_array(d_input, d_output, size);
    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

double cuda_benchmark(void (*func)(const float*, float*, int), const float* input, float* output, int size, int iter) {
    float* d_input;
    float* d_output;
    size_t bytes = size * sizeof(float);
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);
    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++) {
        func(d_input, d_output, size);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceReset();
    return ms / iter;
}