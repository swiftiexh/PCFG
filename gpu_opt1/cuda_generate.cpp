#include <cuda_runtime.h>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <thread>

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

// 异步版接口
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

    // 5. 创建CUDA stream并异步启动kernel
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    kernel_generate<<<gridSize, blockSize, 0, stream>>>(
        d_prefix, prefix_len,
        d_values, MAX_STR_LEN,
        d_results, MAX_RESULT_LEN,
        n
    );

    // 6. CPU可以在这里做其它工作
    std::thread cpu_task([n](){
    // 统计内存占用和时间
    size_t mem = n * MAX_RESULT_LEN;
    std::cout << "[CPU] 预分配结果空间: " << mem / 1024 << " KB" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();

    // 预分配结果空间（模拟，实际分配在后面）
    std::vector<char> pre_alloc(mem, 0);

    // 可以做日志记录
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cout << "[CPU] 预处理耗时: " << elapsed.count() << " 秒" << std::endl;
    });
    cpu_task.join();

    // 7. 等待GPU完成
    cudaStreamSynchronize(stream);

    // 8. 拷贝结果回主机
    std::vector<char> h_results_buf(n * MAX_RESULT_LEN, 0);
    cudaMemcpy(h_results_buf.data(), d_results, n * MAX_RESULT_LEN, cudaMemcpyDeviceToHost);

    // 9. 分配每个结果字符串并赋值给h_results
    for (int i = 0; i < n; ++i) {
        h_results[i] = new char[MAX_RESULT_LEN];
        strncpy(h_results[i], &h_results_buf[i * MAX_RESULT_LEN], MAX_RESULT_LEN - 1);
        h_results[i][MAX_RESULT_LEN - 1] = '\0';
    }

    // 10. 释放device内存和stream
    cudaFree(d_values);
    cudaFree(d_results);
    cudaFree(d_prefix);
    cudaStreamDestroy(stream);
}