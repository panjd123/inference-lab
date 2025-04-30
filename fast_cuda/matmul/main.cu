#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <chrono>
#include <memory>
#include <random>
#include "../MeasurementSeries.h"
#include "../gpu_error.h"
#include "../init.cuh"
#include "matmul.cuh"

template <typename T>
void benchmark(
    size_t M,
    size_t N,
    size_t K,
    int warmup = 1,
    int iterations = 5,
    FAST_MATMUL method = FAST_MATMUL::NATIVE) {
    T* h_A = new T[M * K];
    T* h_B = new T[K * N];
    T* h_C = new T[M * N];
    T* h_C_ref = new T[M * N];
    T* d_A;
    T* d_B;
    T* d_C;
    GPU_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(T)));
    GPU_ERROR(cudaMalloc((void**)&d_B, K * N * sizeof(T)));
    GPU_ERROR(cudaMalloc((void**)&d_C, M * N * sizeof(T)));

    std::mt19937 gen(42);                       // Mersenne Twister 伪随机数生成器
    std::normal_distribution<> dist(0.0, 1.0);  // 均值为0，标准差为1的正态分布
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<T>(dist(gen));
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<T>(dist(gen));
    }
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            T value = 0;
            for (int k = 0; k < K; k++) {
                value += h_A[i * K + k] * h_B[j * K + k];
            }
            h_C_ref[i * N + j] = value;
        }
    }
    cudaMemcpy(d_A, h_A, M * K * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(T), cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) {
        fast_matmul(d_A, d_B, d_C, M, N, K, method);
        GPU_ERROR(cudaGetLastError());
    }
    GPU_ERROR(cudaDeviceSynchronize());
    MeasurementSeries time;
    cudaEvent_t start_event, stop_event;
    for (int i = 0; i < iterations; i++) {
        GPU_ERROR(cudaEventCreate(&start_event));
        GPU_ERROR(cudaEventCreate(&stop_event));
        GPU_ERROR(cudaEventRecord(start_event));
        fast_matmul(d_A, d_B, d_C, M, N, K, method);
        // cudaMemcpy(d_output, d_input, N * sizeof(T), cudaMemcpyDeviceToDevice);
        GPU_ERROR(cudaEventRecord(stop_event));
        GPU_ERROR(cudaEventSynchronize(stop_event));
        float elapsed_time;
        GPU_ERROR(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
        time.add(elapsed_time);
    }
    double milliseconds = time.median();
    double avg_seconds = milliseconds / 1000.0;

    double memory_transfer = 0;  // in bytes
    if (method == FAST_MATMUL::NATIVE) {
        memory_transfer = (M * N * (2 * K + 1)) * sizeof(T);
    } else if (method == FAST_MATMUL::SHARED_MEMORY) {
        memory_transfer = (M * N + M * K * ceil_div(N, 16) + K * N * ceil_div(M, 16)) * sizeof(T);
    }

    double bandwidth = memory_transfer / (avg_seconds * 1024.0 * 1024.0 * 1024.0);  // in GB/s
    double gflops = (2 * M * N * K) / (avg_seconds * 1024.0 * 1024.0 * 1024.0);     // in GFLOPS
    printf("%20ld%20.8f%20.8f%20.8f%20.8f%50s\n", N, avg_seconds * 1000.0, memory_transfer / (1024.0 * 1024.0), bandwidth, gflops, fast_matmul_method_to_string(method).c_str());
    cudaMemcpy(h_C, d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) / (fabs(h_C[i]) + fabs(h_C_ref[i]) + 1e-7) > 1e-3 && fabs(h_C[i] - h_C_ref[i]) > 1e-4) {
            printf("Error: %.10f != %.10f\n", h_C[i], h_C_ref[i]);
            break;
        }
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

void cuda_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  L2 cache size: %.2f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Max registers per block: %d\n", prop.regsPerBlock);
        printf("  Max Shared memory per multiprocessor: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    }
    printf("\n");
}

int main() {
    cuda_info();
    printf("%20s%20s%20s%20s%20s%50s\n", "N", "Time (ms)", "Memory (MB)", "Bandwidth (GB/s)", "GFLOPS", "Method");
    for (size_t n = 256; n <= 4096; n *= 2) {
        for (FAST_MATMUL method : {FAST_MATMUL::NATIVE, FAST_MATMUL::SHARED_MEMORY}) {  // , FAST_MATMUL::SHARED_MEMORY
            benchmark<float>(n, n, n, 1, 5, method);
        }
    }
    return 0;
}