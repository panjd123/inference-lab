#include <cuda_runtime.h>
#include <string>

enum class FAST_MATMUL {
    NATIVE,
    SHARED_MEMORY,
};

std::string fast_matmul_method_to_string(FAST_MATMUL method) {
    switch (method) {
        case FAST_MATMUL::NATIVE:
            return "NATIVE";
        case FAST_MATMUL::SHARED_MEMORY:
            return "SHARED_MEMORY";
        default:
            return "UNKNOWN";
    }
}

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

template <int M, int N, int K>
struct GemmShape {
    static const int kM = M;
    static const int kN = N;
    static const int kK = K;
    static const int kMN = M * N;
    static const int kMK = M * K;
    static const int kNK = N * K;
    static const int kCount = M * N * K;
};

struct GemmCoord {
    int m;
    int n;
    int k;
    __host__ __device__ GemmCoord(int m, int n, int k) : m(m), n(n), k(k) {}
};

template <int Row, int Col>
struct MatrixShape {
    static const int kRow = Row;
    static const int kCol = Col;
    static const int kCount = Row * Col;
};

struct MatrixCoord {
    int row;
    int col;
    __host__ __device__ MatrixCoord(int r, int c) : row(r), col(c) {}
};

template <typename ThreadblockShape>
__host__ __device__ MatrixCoord get_threadblock_offset(int block_idx, GemmCoord gemm_coord) {
    int n_tile = (gemm_coord.n + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    int threadblock_offset_row = (block_idx / n_tile) * ThreadblockShape::kM;
    int threadblock_offset_col = (block_idx % n_tile) * ThreadblockShape::kN;
    return MatrixCoord(threadblock_offset_row, threadblock_offset_col);
}

template <typename T, typename ThreadblockShape>
__global__ void fast_matmul_kernel_shared_memory(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ T shared_A[ThreadblockShape::kM][ThreadblockShape::kK];
    __shared__ T shared_B[ThreadblockShape::kN][ThreadblockShape::kK];

    int row = threadIdx.x / ThreadblockShape::kN;
    int col = threadIdx.x % ThreadblockShape::kN;
    MatrixCoord threadblock_offset = get_threadblock_offset<ThreadblockShape>(blockIdx.x, GemmCoord(M, N, K));
    int threadblock_offset_row = threadblock_offset.row;
    int threadblock_offset_col = threadblock_offset.col;

    T value = 0;

    int kIterations = (K + 15) / 16;

    for (int k = 0; k < kIterations; ++k) {
        int threadblock_offset_k = k * 16;
        {
            int row = threadIdx.x / ThreadblockShape::kN;
            int col = threadIdx.x % ThreadblockShape::kN;
            if (threadblock_offset_row + row < M && threadblock_offset_k + col < K) {
                shared_A[row][col] = A[(threadblock_offset_row + row) * K + threadblock_offset_k + col];
            } else {
                shared_A[row][col] = 0;
            }
        }
        {
            int row = threadIdx.x % ThreadblockShape::kN;
            int col = threadIdx.x / ThreadblockShape::kN;
            if (threadblock_offset_col + col < N && threadblock_offset_k + row < K) {
                shared_B[col][row] = B[(threadblock_offset_col + col) * K + threadblock_offset_k + row];
            } else {
                shared_B[col][row] = 0;
            }
        }
        __syncthreads();
        for (int i = 0; i < ThreadblockShape::kK; ++i) {
            value += shared_A[row][i] * shared_B[col][i];
        }
        __syncthreads();
    }
    C[(threadblock_offset_row + row) * N + (threadblock_offset_col + col)] = value;
}

template <typename T>
__global__ void fast_matmul_kernel_native(const T* A, const T* B, T* C, int M, int N, int K) {
    // Native CUDA implementation of matrix multiplication
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T value = 0;
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = value;
    }
}

template <typename T>
void fast_matmul(const T* A, const T* B, T* C, int M, int N, int K, FAST_MATMUL method = FAST_MATMUL::NATIVE) {
    // Implementation of matrix multiplication using the specified method
    // This is a placeholder implementation. The actual implementation will depend on the chosen method.
    if (method == FAST_MATMUL::NATIVE) {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        fast_matmul_kernel_native<<<grid, block>>>(A, B, C, M, N, K);
    } else if (method == FAST_MATMUL::SHARED_MEMORY) {
        using ThreadblockShape = GemmShape<16, 16, 16>;  // Example threadblock shape
        dim3 block(ThreadblockShape::kM * ThreadblockShape::kN);
        dim3 grid(ceil_div(M, ThreadblockShape::kM) * ceil_div(N, ThreadblockShape::kN));
        fast_matmul_kernel_shared_memory<T, ThreadblockShape><<<grid, block>>>(A, B, C, M, N, K);
    }
}