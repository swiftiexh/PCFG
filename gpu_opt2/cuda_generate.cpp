#include <cuda_runtime.h>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>

#define MAX_STR_LEN 128
#define MAX_RESULT_LEN 256

// CUDA核函数：每个线程拼接一个猜测
__global__ void kernel_generate(
    const char* prefix, int prefix_len,
    const char* values, int value_stride,
    char* results, int result_stride,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const char* value_ptr = values + idx * value_stride;
        char* result_ptr = results + idx * result_stride;

        // 拷贝prefix
        for (int i = 0; i < prefix_len; ++i)
            result_ptr[i] = prefix[i];

        // 拷贝value
        int vlen = 0;
        for (; vlen < value_stride && value_ptr[vlen] != '\0'; ++vlen)
            result_ptr[prefix_len + vlen] = value_ptr[vlen];

        // 结尾
        result_ptr[prefix_len + vlen] = '\0';
    }
}

// 对外接口
extern "C" void cuda_generate_guesses(
    const char* prefix,
    char** h_values,
    int n,
    char** h_results)
{
    // 1. 计算prefix长度
    int prefix_len = strlen(prefix);

    // 2. 分配并拷贝所有value到连续的device内存
    std::vector<char> h_values_buf(n * MAX_STR_LEN, 0);
    for (int i = 0; i < n; ++i) {
        strncpy(&h_values_buf[i * MAX_STR_LEN], h_values[i], MAX_STR_LEN - 1);
    }

    char* d_values = nullptr;
    cudaMalloc(&d_values, n * MAX_STR_LEN);
    cudaMemcpy(d_values, h_values_buf.data(), n * MAX_STR_LEN, cudaMemcpyHostToDevice);

    // 3. 分配结果空间
    char* d_results = nullptr;
    cudaMalloc(&d_results, n * MAX_RESULT_LEN);

    // 4. 拷贝prefix到device
    char* d_prefix = nullptr;
    cudaMalloc(&d_prefix, MAX_STR_LEN);
    cudaMemcpy(d_prefix, prefix, prefix_len + 1, cudaMemcpyHostToDevice);

    // 5. 启动kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    kernel_generate<<<gridSize, blockSize>>>(
        d_prefix, prefix_len,
        d_values, MAX_STR_LEN,
        d_results, MAX_RESULT_LEN,
        n
    );
    cudaDeviceSynchronize();

    // 6. 拷贝结果回主机
    std::vector<char> h_results_buf(n * MAX_RESULT_LEN, 0);
    cudaMemcpy(h_results_buf.data(), d_results, n * MAX_RESULT_LEN, cudaMemcpyDeviceToHost);

    // 7. 分配每个结果字符串并赋值给h_results
    for (int i = 0; i < n; ++i) {
        h_results[i] = new char[MAX_RESULT_LEN];
        strncpy(h_results[i], &h_results_buf[i * MAX_RESULT_LEN], MAX_RESULT_LEN - 1);
        h_results[i][MAX_RESULT_LEN - 1] = '\0';
    }

    // 8. 释放device内存
    cudaFree(d_values);
    cudaFree(d_results);
    cudaFree(d_prefix);
}