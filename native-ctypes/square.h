#pragma once
// 声明一个 C 接口函数，用于计算数组平方
extern "C" void square_array(const float* input, float* output, int size);
extern "C" void parallel_square_array(const float* input, float* output, int size);
extern "C" void native_cuda_square_array(const float* input, float* output, int size);
extern "C" void cuda_square_array(const float* input, float* output, int size);
extern "C" void our_parallel_square_array(const float* input, float* output, int size);