#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 检查 CUDA API 调用的返回值，失败则打印文件/行号并退出
#define CUDA_CHECK(cmd)                                                   \
  do {                                                                    \
    cudaError_t err_ = (cmd);                                             \
    if (err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,    \
                   __LINE__, cudaGetErrorString(err_), #cmd);             \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

// kernel <<<>>> launch 本身不返回错误码，launch 之后调用此宏捕获
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
