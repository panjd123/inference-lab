#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <format>
#include <iostream>

template <int CubeRootIterations>
inline float native_newton_cube_root(float x) {
    float guess = x / 3.0f;
#pragma GCC unroll 1
    for (int i = 0; i < CubeRootIterations; ++i) {
        guess = (2.0f * guess + x / (guess * guess)) / 3.0f;
    }
    return guess;
}

// 编译期常量迭代 Newton 法
template <int CubeRootIterations>
inline float newton_cube_root(float x) {
    //   float guess = x / 3.0f;
    float guess = x * (1 / 3.0f);
    for (int i = 0; i < CubeRootIterations; ++i) {
        guess = (2.0f * guess + x / (guess * guess)) / 3.0f;
    }
    return guess;
}

template <int CubeRootIterations>
inline void avx2_newton_cube_root(const float* input, float* output) {
    __m256 x = _mm256_loadu_ps(input);
    __m256 guess = _mm256_mul_ps(x, _mm256_set1_ps(1.0f / 3.0f));
    for (int i = 0; i < CubeRootIterations; ++i) {
        __m256 x_div_guess2 = _mm256_div_ps(x, _mm256_mul_ps(guess, guess));
        guess = _mm256_div_ps(
            _mm256_add_ps(_mm256_mul_ps(guess, _mm256_set1_ps(2.0f)), x_div_guess2),
            _mm256_set1_ps(3.0f));
    }
    _mm256_storeu_ps(output, guess);
}

template <int CubeRootIterations>
void native_cube_root_template(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = native_newton_cube_root<CubeRootIterations>(input[i]);
    }
}

// 工厂函数模板，返回函数指针
template <int CubeRootIterations>
void opt_cube_root_template(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = newton_cube_root<CubeRootIterations>(input[i]);
    }
}

template <int CubeRootIterations>
void avx2_cube_root_template(const float* input, float* output, int size) {
    size = (size + 7) / 8 * 8;  // 向上取整到8的倍数
    for (int i = 0; i < size; i += 8) {
        avx2_newton_cube_root<CubeRootIterations>(input + i, output + i);
    }
    //   for (int i = size - size % 8; i < size; ++i) {
    //     output[i] = newton_cube_root<CubeRootIterations>(input[i]);
    //   }
}

template <int CubeRootIterations>
void parallel_cube_root_template(const float* input, float* output, int size) {
#pragma omp parallel for private(input, output, size)
    for (int i = 0; i < size; ++i) {
        output[i] = newton_cube_root<CubeRootIterations>(input[i]);
    }
}

template <int CubeRootIterations>
void parallel_avx2_cube_root_template(const float* input,
                                      float* output,
                                      int size) {
    size = (size + 7) / 8 * 8;  // 向上取整到8的倍数
#pragma omp parallel for private(input, output, size)
    for (int i = 0; i < size; i += 8) {
        avx2_newton_cube_root<CubeRootIterations>(input + i, output + i);
    }
    // for (int i = size - size % 8; i < size; ++i) {
    //     output[i] = newton_cube_root<CubeRootIterations>(input[i]);
    // }
}

template <int CubeRootIterations>
void dynamic_avx2_parallel_cube_root_template(const float* input,
                                              float* output,
                                              int size) {
    size = (size + 7) / 8 * 8;  // 向上取整到8的倍数
#pragma omp parallel for schedule(dynamic, 1000) private(input, output, size)
    for (int i = 0; i < size; i += 8) {
        avx2_newton_cube_root<CubeRootIterations>(input + i, output + i);
    }
}

double performance_cpu_weight = 2;
double efficient_cpu_weight = 0;

template <int CubeRootIterations>
void manual_avx2_parallel_cube_root_template(const float* input,
                                             float* output,
                                             int size) {
    double weight_sum = performance_cpu_weight * 16 + efficient_cpu_weight * 16;
    double pieces_size = size / weight_sum;
    int performance_piece_size = performance_cpu_weight * pieces_size;
    int efficient_piece_size = efficient_cpu_weight * pieces_size;
    int residual = size - performance_piece_size * 16 - efficient_piece_size * 16;
    if (residual > 32) {
        printf("too large residual %d\n", residual);
    }
    int workspace_offset[33] = {0};
    for (int i = 0; i < 16; ++i) {
        workspace_offset[i] = performance_piece_size;
    }
    for (int i = 16; i < 32; ++i) {
        workspace_offset[i] = efficient_piece_size;
    }
    for (int i = 0; i < residual; ++i) {
        workspace_offset[i] += 1;
    }
    workspace_offset[32] = 0;
    for (int i = 1; i < 33; ++i) {
        workspace_offset[i] += workspace_offset[i - 1];
    }
    if (workspace_offset[32] != size) {
        printf("workspace_offset[32] %d != size %d\n", workspace_offset[32], size);
    }
#pragma omp parallel private(input, output, size)
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();  // 返回当前线程正在运行的 CPU core ID
        // printf("Thread %d is running on core %d\n", tid, cpu);
        int start, end;
        if (nt != 32) {
            int workspace_size = (size + nt - 1) / nt;
            start = tid * workspace_size;
            end = std::min(start + workspace_size, size);
        } else {
            start = workspace_offset[cpu];
            end = workspace_offset[cpu + 1];
            // printf("%d %d %d %d\n", cpu, start, end, end - start);
        }
        int i = start;
        for (i = start; i < end; i += 8) {
            avx2_newton_cube_root<CubeRootIterations>(input + i,
                                                      output + i);
        }
        for (int j = i; j < end; ++j) {
            output[j] = newton_cube_root<CubeRootIterations>(input[j]);
        }
    }
}

// 性能测试函数
double benchmark(void (*func)(const float*, float*, int),
                 const float* input,
                 float* output,
                 int size,
                 int iter,
                 int omp_threads = -1) {
    if (omp_threads > 0) {
        omp_set_num_threads(omp_threads);
    }
    for (int i = 0; i < iter / 10; ++i) {
        func(input, output, size);
    }
    auto start = std::chrono::high_resolution_clock::now();
    volatile float sink = 0.0f;
    for (int i = 0; i < iter; ++i) {
        func(input, output, size);
        sink += output[0];
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    omp_set_num_threads(omp_get_max_threads());
    // for (int i = 0; i < size; ++i) {
    //     float grond_truth = newton_cube_root<1000>(input[i]);
    //     if (std::abs(output[i] - grond_truth) > 1e-3) {
    //         std::cout << "output[" << i << "] = " << output[i]
    //                   << " is not equal to ground truth " << grond_truth
    //                   << std::endl;
    //         return -1;
    //     }
    // }
    return ms / iter;
}

template <int... Iterations>
void run_all_configs(const float* input,
                     float* output,
                     int N,
                     int iter,
                     float float_op) {
    (
        [&] {
            int ci = Iterations;

            int n = float_op / (iter * ci);
            if (n > N) {
                std::cout << "n is too large " << n << std::endl;
                return;
            }

            auto ms_native = benchmark(native_cube_root_template<Iterations>, input,
                                       output, n, iter);
            auto ms_opt = benchmark(opt_cube_root_template<Iterations>, input,
                                    output, n, iter);
            auto ms_avx2 = benchmark(avx2_cube_root_template<Iterations>, input,
                                     output, n, iter);
            auto ms_parallel = benchmark(parallel_cube_root_template<Iterations>,
                                         input, output, n, iter);
            auto ms_dynamic =
                benchmark(dynamic_avx2_parallel_cube_root_template<Iterations>, input,
                          output, n, iter);
            double ms_parallel_avx2 =
                benchmark(parallel_avx2_cube_root_template<Iterations>, input,
                          output, n, iter);
            double ms_manual =
                benchmark(manual_avx2_parallel_cube_root_template<Iterations>, input,
                          output, n, iter);
            double iterps_native = ci * n / ms_native * 1000;
            double iterps_opt = ci * n / ms_opt * 1000;
            double iterps_avx2 = ci * n / ms_avx2 * 1000;
            double iterps_parallel = ci * n / ms_parallel * 1000;
            double iterps_dynamic = ci * n / ms_dynamic * 1000;
            double iterps_parallel_avx2 = ci * n / ms_parallel_avx2 * 1000;
            double iterps_manual = ci * n / ms_manual * 1000;
            std::cout << std::format("{:>8} {:>8} {:>6} {:>20} {:>10f} {:>10g}", ci, n,
                                     n * sizeof(float) * 2 / 1024, "Native",
                                     ms_native, iterps_native / 1e6)
                      << std::endl;
            std::cout << std::format("{:>8} {:>8} {:>6} {:>20} {:>10f} {:>10g}", ci, n,
                                     n * sizeof(float) * 2 / 1024, "Opt",
                                     ms_opt, iterps_opt / 1e6)
                      << std::endl;
            std::cout << std::format("{:>8} {:>8} {:>6} {:>20} {:>10f} {:>10g}", ci, n,
                                     n * sizeof(float) * 2 / 1024, "AVX2", ms_avx2,
                                     iterps_avx2 / 1e6)
                      << std::endl;
            std::cout << std::format("{:>8} {:>8} {:>6} {:>20} {:>10f} {:>10g}", ci, n,
                                     n * sizeof(float) * 2 / 1024, "Parallel",
                                     ms_parallel, iterps_parallel / 1e6)
                      << std::endl;
            std::cout << std::format("{:>8} {:>8} {:>6} {:>20} {:>10f} {:>10g}", ci, n,
                                     n * sizeof(float) * 2 / 1024, "Parallel_AVX2",
                                     ms_parallel_avx2, iterps_parallel_avx2 / 1e6)
                      << std::endl;
            std::cout << std::format("{:>8} {:>8} {:>6} {:>20} {:>10f} {:>10g}", ci, n,
                                     n * sizeof(float) * 2 / 1024, "Dynamic_AVX2",
                                     ms_dynamic, iterps_dynamic / 1e6)
                      << std::endl;
            std::cout << std::format("{:>8} {:>8} {:>6} {:>20} {:>10f} {:>10g}", ci, n,
                                     n * sizeof(float) * 2 / 1024, "Manual",
                                     ms_manual, iterps_manual / 1e6)
                      << std::endl;
        }(),
        ...);  // fold expression over Iterations
}

// 主函数
int main() {
    int N = 1048576;
    float* input = new float[N];
    float* output = new float[N];
    for (int i = 0; i < N; ++i) {
        input[i] = static_cast<float>(i);
    }

    const float float_op = 1e8;
    const int iter = 100;

    std::cout << std::format("{:>8} {:>8} {:>6} {:>20} {:>10} {:>10}", "newton",
                             "size", "KB", "algo", "ms", "MIter/s")
              << std::endl;
    // run_all_configs<1, 2, 3, 10, 14, 15, 16, 17, 18, 50, 100, 200, 500,
    //                 1000>(input, output, N, iter, float_op);
    run_all_configs<2, 16, 18, 100, 1000, 10000>(input, output, N, iter, float_op);

    // 反常多线程
    // run_all_configs<3>(input, output, N, iter, float_op);

    delete[] input;
    delete[] output;
}
