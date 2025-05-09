#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <chrono>
#include <memory>
#include "../MeasurementSeries.h"
#include "../gpu_error.h"
#include "../init.cuh"
#include "square.cuh"

template <typename T>
void benchmark(
    size_t N,
    int warmup = 1,
    int iterations = 5,
    FAST_SQUARE method = FAST_SQUARE::NATIVE) {
    T* d_input;
    T* d_output;
    GPU_ERROR(cudaMalloc((void**)&d_input, N * sizeof(T)));
    GPU_ERROR(cudaMalloc((void**)&d_output, N * sizeof(T)));
    for (int i = 0; i < warmup; i++) {
        fast_square(d_input, d_output, N, method);
        GPU_ERROR(cudaGetLastError());
    }
    GPU_ERROR(cudaDeviceSynchronize());
    MeasurementSeries time;
    cudaEvent_t start_event, stop_event;
    for (int i = 0; i < iterations; i++) {
        GPU_ERROR(cudaEventCreate(&start_event));
        GPU_ERROR(cudaEventCreate(&stop_event));
        GPU_ERROR(cudaEventRecord(start_event));
        fast_square(d_input, d_output, N, method);
        // cudaMemcpy(d_output, d_input, N * sizeof(T), cudaMemcpyDeviceToDevice);
        GPU_ERROR(cudaEventRecord(stop_event));
        GPU_ERROR(cudaEventSynchronize(stop_event));
        float elapsed_time;
        GPU_ERROR(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
        time.add(elapsed_time);
    }
    double milliseconds = time.median();
    double avg_seconds = milliseconds / 1000.0;
    double memory_transfer = N * sizeof(T) * 2;                                     // in bytes
    double bandwidth = memory_transfer / (avg_seconds * 1024.0 * 1024.0 * 1024.0);  // in GB/s
    double gflops = (N) / (avg_seconds * 1024.0 * 1024.0 * 1024.0);                 // in GFLOPS
    printf("%20ld%20.8f%20.8f%20.8f%20.8f%50s\n", N, avg_seconds * 1000.0, memory_transfer / (1024.0 * 1024.0), bandwidth, gflops, fast_square_method_to_string(method).c_str());

    cudaFree(d_input);
    cudaFree(d_output);
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
    for (size_t n = 1024; n <= 1024 * 1024 * 1024; n *= 4) {
        for (FAST_SQUARE method : {FAST_SQUARE::NATIVE, FAST_SQUARE::GLOBAL_MEMORY_ACCESS_COALESCE}) {
            benchmark<float>(n, 5, 100, method);
        }
    }
    return 0;
}

// https://cuda.godbolt.org/z/d8cMrjsoE