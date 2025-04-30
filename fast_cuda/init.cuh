#include <cuda_runtime.h>

template <typename T>
void init(T* input, int size) {
    int bytes_size = size * sizeof(T);
    uint8_t* h_input = new uint8_t[bytes_size];
    for (int i = 0; i < size; i++) {
        h_input[i] = i % 256;
    }
    cudaMemcpy(input, h_input, bytes_size, cudaMemcpyHostToDevice);
    delete[] h_input;
}