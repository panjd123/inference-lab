#include <cuda_runtime.h>
#include <string>
#include "matmul.h"

__host__ __device__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

struct MatrixCoord {
    int row;
    int col;
    __host__ __device__ MatrixCoord(int r, int c) : row(r), col(c) {}
};

template <int Row, int Col>
struct MatrixShape {
    static const int kRow = Row;
    static const int kCol = Col;
    static const int kCount = Row * Col;
    __host__ __device__ static MatrixCoord to_matrix_coord() {
        return MatrixCoord(Row, Col);
    }
};

struct PitchLinearCoord {
    int continuous;
    int stride;
    __host__ __device__ PitchLinearCoord(int c, int s) : continuous(c), stride(s) {}
};

template <int Continuous, int Stride>
struct PitchLinearShape {
    static const int kContinuous = Continuous;
    static const int kStride = Stride;
    static const int kCount = Continuous * Stride;
    __host__ __device__ static PitchLinearCoord to_pitch_linear_coord() {
        return PitchLinearCoord(Continuous, Stride);
    }
};

struct GemmCoord {
    int m;
    int n;
    int k;
    __host__ __device__ GemmCoord(int m, int n, int k) : m(m), n(n), k(k) {}
    __host__ __device__ MatrixCoord mn() const {
        return MatrixCoord(m, n);
    }
    __host__ __device__ MatrixCoord mk() const {
        return MatrixCoord(m, k);
    }
    __host__ __device__ MatrixCoord nk() const {
        return MatrixCoord(n, k);
    }
};

template <int M, int N, int K>
struct GemmShape {
    static const int kM = M;
    static const int kN = N;
    static const int kK = K;
    static const int kMN = M * N;
    static const int kMK = M * K;
    static const int kNK = N * K;
    static const int kCount = M * N * K;
    using MN = MatrixShape<M, N>;
    using MK = MatrixShape<M, K>;
    using NK = MatrixShape<N, K>;
    static __host__ __device__ GemmCoord to_gemm_coord() {
        return GemmCoord(M, N, K);
    }
};

template <typename ThreadblockShape>
__host__ __device__ MatrixCoord get_threadblock_offset(int block_idx, GemmCoord gemm_coord) {
    int n_tile = (gemm_coord.n + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    int threadblock_offset_row = (block_idx / n_tile) * ThreadblockShape::kM;
    int threadblock_offset_col = (block_idx % n_tile) * ThreadblockShape::kN;
    return MatrixCoord(threadblock_offset_row, threadblock_offset_col);
}

template <typename ThreadblockShape, typename WarpShape>
__host__ __device__ MatrixCoord get_warp_offset(int thread_idx) {
    using WarpCount = MatrixShape<ThreadblockShape::kM / WarpShape::kM, ThreadblockShape::kN / WarpShape::kN>;
    int warp_idx = thread_idx / 32;
    int warp_row = warp_idx / WarpCount::kCol;
    int warp_col = warp_idx % WarpCount::kCol;
    return MatrixCoord(warp_row * WarpShape::kM, warp_col * WarpShape::kN);
}

template <FAST_MATMUL method>
struct DefaultParams {
    // using ThreadblockShape = GemmShape<32, 32, 32>;
    // using WarpShape = GemmShape<16, 16, 32>;
    using ThreadblockShape = GemmShape<16, 16, 16>;
    using WarpShape = GemmShape<16, 16, 16>;
};

// template <>
// struct DefaultParams<FAST_MATMUL::SHARED_MEMORY> {
//     using ThreadblockShape = GemmShape<32, 32, 32>;
// };

// template <>
// struct DefaultParams<FAST_MATMUL::SHARED_MEMORY_NATIVE> {
//     using ThreadblockShape = GemmShape<32, 32, 32>;
// };

struct DefaultArgument {
    GemmCoord threadblock_shape;
    GemmCoord warp_shape;

    DefaultArgument() : threadblock_shape(32, 32, 32), warp_shape(16, 16, 32) {}
};

DefaultArgument get_default_args(FAST_MATMUL method) {
    DefaultArgument default_args;
    switch (method) {
        case FAST_MATMUL::NATIVE:
            default_args.threadblock_shape = DefaultParams<FAST_MATMUL::NATIVE>::ThreadblockShape::to_gemm_coord();
            default_args.warp_shape = DefaultParams<FAST_MATMUL::NATIVE>::WarpShape::to_gemm_coord();
            break;
        case FAST_MATMUL::SHARED_MEMORY:
            default_args.threadblock_shape = DefaultParams<FAST_MATMUL::SHARED_MEMORY>::ThreadblockShape::to_gemm_coord();
            default_args.warp_shape = DefaultParams<FAST_MATMUL::SHARED_MEMORY>::WarpShape::to_gemm_coord();
            break;
        case FAST_MATMUL::SHARED_MEMORY_NATIVE:
            default_args.threadblock_shape = DefaultParams<FAST_MATMUL::SHARED_MEMORY_NATIVE>::ThreadblockShape::to_gemm_coord();
            default_args.warp_shape = DefaultParams<FAST_MATMUL::SHARED_MEMORY_NATIVE>::WarpShape::to_gemm_coord();
            break;
        default:
            break;
    }
    return default_args;
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

    int kIterations = ceil_div(K, ThreadblockShape::kK);

    for (int k = 0; k < kIterations; ++k) {
        int threadblock_offset_k = k * ThreadblockShape::kK;
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

// template __global__ void fast_matmul_kernel_shared_memory<float, GemmShape<16, 16, 16>>(
//     const float* A,
//     const float* B,
//     float* C,
//     int M,
//     int N,
//     int K);

template <typename T, typename ThreadblockShape>
__global__ void fast_matmul_kernel_shared_memory_native(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ T shared_A[ThreadblockShape::kM][ThreadblockShape::kK];
    __shared__ T shared_B[ThreadblockShape::kN][ThreadblockShape::kK];

    int row = threadIdx.x / ThreadblockShape::kN;
    int col = threadIdx.x % ThreadblockShape::kN;
    MatrixCoord threadblock_offset = get_threadblock_offset<ThreadblockShape>(blockIdx.x, GemmCoord(M, N, K));
    int threadblock_offset_row = threadblock_offset.row;
    int threadblock_offset_col = threadblock_offset.col;

    T value = 0;

    int kIterations = ceil_div(K, ThreadblockShape::kK);

    for (int k = 0; k < kIterations; ++k) {
        int threadblock_offset_k = k * ThreadblockShape::kK;
        {
            if (threadblock_offset_row + row < M && threadblock_offset_k + col < K) {
                shared_A[row][col] = A[(threadblock_offset_row + row) * K + threadblock_offset_k + col];
            } else {
                shared_A[row][col] = 0;
            }
        }
        {
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
    // nvcuda::wmma::load_matrix_sync
}

template <typename T, typename ThreadblockShape, typename WarpShape, int isMatrixA = 1>
class G2SIterator {
    // 128x128
    using WarpCount = MatrixShape<ThreadblockShape::kM / WarpShape::kM, ThreadblockShape::kN / WarpShape::kN>;
    // 4x2
    using Shape = std::conditional_t<
        isMatrixA,
        PitchLinearShape<ThreadblockShape::kK, ThreadblockShape::kM>,
        PitchLinearShape<ThreadblockShape::kK, ThreadblockShape::kN>>;
    static const int kWarpSize = 32;
    static const int kThreadCount = WarpCount::kCount * kWarpSize;                                // 256
    static const int kAccessElements = 128 / (sizeof(T) * 8);                                     // 8
    using kAccessCount = PitchLinearShape<Shape::kContinuous / kAccessElements, Shape::kStride>;  // 16, 128
    static const int kIterations = kAccessCount::kCount / kThreadCount;                           // 8
    // 16, 16

    uint8_t* src_ptr;
    uint8_t* dst_ptr;
    int global_delta;
    int global_delta_stage;
    int smem_delta;
    int smem_delta_stage;

    __host__ __device__ G2SIterator(T* src, T* dst, GemmCoord extent, MatrixCoord threadblock_offset, int thread_idx) {
        int threadblock_offset_stride = isMatrixA ? threadblock_offset.row : threadblock_offset.col;
        int extent_continuous = isMatrixA ? extent.m : extent.n;
        int thread_offset_continuous = thread_idx % kAccessCount::kContinuous;
        int thread_offset_stride = thread_idx / kAccessCount::kContinuous;
        src_ptr = reinterpret_cast<uint8_t*>(src) +
                  (threadblock_offset_stride + thread_offset_stride) * extent.k +
                  thread_offset_continuous * sizeof(T);
        dst_ptr = reinterpret_cast<uint8_t*>(dst) + thread_offset_stride * Shape::kStride +
                  thread_offset_continuous * sizeof(T);
        global_delta = extent_continuous * sizeof(T) * Shape::kStride / kIterations;
        /*
        x . . . . . .   delta_stage  x . . . . . .
        . . . . . . .
        . . . . . . .
        delta
        x . . . . . .
        . . . . . . .
        */
        global_delta_stage = -kIterations * global_delta + ThreadblockShape::kK * sizeof(T);
        smem_delta = Shape::kCount * sizeof(T) / kIterations;
        smem_delta_stage = -kIterations * smem_delta;
    }

    __host__ __device__ void load() {
        for (int i = 0; i < kIterations; ++i) {
            *reinterpret_cast<int4*>(dst_ptr) = *reinterpret_cast<int4*>(src_ptr);
            src_ptr += global_delta;
            dst_ptr += smem_delta;
        }
        src_ptr += global_delta_stage;
        dst_ptr += smem_delta_stage;
    }
};

template <typename T, typename WarpShape, typename InstructionShape>
class Mma {
};

template <typename T, typename ThreadblockShape, typename WarpShape, typename InstructionShape>
class FastMatmulTile {
    __host__ __device__ FastMatmulTile() {}
    __device__ void fast_matmul_kernel_tile(const T* A, const T* B, T* C, int M, int N, int K) {
        __shared__ T shared_A[ThreadblockShape::kM][ThreadblockShape::kK];
        __shared__ T shared_B[ThreadblockShape::kN][ThreadblockShape::kK];

        MatrixCoord threadblock_offset = get_threadblock_offset<ThreadblockShape>(blockIdx.x, GemmCoord(M, N, K));
        MatrixCoord warp_offset = get_threadblock_offset<WarpShape>(blockIdx.x, GemmCoord(M, N, K));
        int threadblock_offset_row = threadblock_offset.row;
        int threadblock_offset_col = threadblock_offset.col;

        int kIterations = ceil_div(K, ThreadblockShape::kK);

        G2SIterator<T, ThreadblockShape, WarpShape> g2s_A(A, shared_A, GemmCoord(M, N, K), threadblock_offset, threadIdx.x);
        G2SIterator<T, ThreadblockShape, WarpShape, 0> g2s_B(B, shared_B, GemmCoord(M, N, K), threadblock_offset, threadIdx.x);

        for (int k = 0; k < kIterations; ++k) {
            int threadblock_offset_k = k * ThreadblockShape::kK;
            g2s_A.load();
            g2s_B.load();
        }
    }
};

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
        using DefaultParams = DefaultParams<FAST_MATMUL::SHARED_MEMORY>;
        using ThreadblockShape = DefaultParams::ThreadblockShape;
        dim3 block(ThreadblockShape::kM * ThreadblockShape::kN);
        dim3 grid(ceil_div(M, ThreadblockShape::kM) * ceil_div(N, ThreadblockShape::kN));
        fast_matmul_kernel_shared_memory<T, ThreadblockShape><<<grid, block>>>(A, B, C, M, N, K);
    } else if (method == FAST_MATMUL::SHARED_MEMORY_NATIVE) {
        using DefaultParams = DefaultParams<FAST_MATMUL::SHARED_MEMORY_NATIVE>;
        using ThreadblockShape = DefaultParams::ThreadblockShape;
        dim3 block(ThreadblockShape::kM * ThreadblockShape::kN);
        dim3 grid(ceil_div(M, ThreadblockShape::kM) * ceil_div(N, ThreadblockShape::kN));
        fast_matmul_kernel_shared_memory_native<T, ThreadblockShape><<<grid, block>>>(A, B, C, M, N, K);
    }
}