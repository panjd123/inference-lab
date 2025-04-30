enum class FAST_SQUARE {
    NATIVE,
    GLOBAL_MEMORY_ACCESS_COALESCE,
};

std::string fast_square_method_to_string(FAST_SQUARE method) {
    switch (method) {
        case FAST_SQUARE::NATIVE:
            return "NATIVE";
        case FAST_SQUARE::GLOBAL_MEMORY_ACCESS_COALESCE:
            return "GLOBAL_MEMORY_ACCESS_COALESCE";
        default:
            return "UNKNOWN";
    }
}

template <typename T>
__global__ void fast_square_kernel_native(T* input, T* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
};

template <typename T>
__global__ void fast_square_kernel_global_memory_access_coalesce(int4* input, int4* output, size_t size) {
    static const int kNelement = sizeof(int4) / sizeof(T);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int4 input_int4 = input[idx];
        int4 output_int4;
        T* input_ptr = (T*)&input_int4;
        T* output_ptr = (T*)&output_int4;
#pragma unroll
        for (int i = 0; i < kNelement; i++) {
            output_ptr[i] = input_ptr[i] * input_ptr[i];
        }
        output[idx] = output_int4;
    }
};

template <typename T>
void fast_square(T* input, T* output, size_t size, FAST_SQUARE method = FAST_SQUARE::NATIVE);

template <typename T>
void fast_square<T, FAST_SQUARE::NATIVE>(T* input, T* output, size_t size, FAST_SQUARE method) {
    static const int block_size = 256;
    if (method == FAST_SQUARE::GLOBAL_MEMORY_ACCESS_COALESCE) {
        int int4_size = size * sizeof(T) / sizeof(int4);
        fast_square_kernel_global_memory_access_coalesce<T><<<(int4_size + block_size - 1) / block_size, block_size>>>((int4*)input, (int4*)output, int4_size);
    } else if (method == FAST_SQUARE::NATIVE) {
        fast_square_kernel_native<<<(size + block_size - 1) / block_size, block_size>>>(input, output, size);
    }
}