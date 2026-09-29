#pragma once

#include <cuda_runtime.h>
#include <random>
#include <vector>
#include "common/errors.h"

namespace hopper {

// host 端均匀分布随机向量。固定 seed 保证实验可复现。
// （后续 fp16/fp8 实验在此基础上加特化）
inline std::vector<float> make_random_vector(size_t n, uint32_t seed = 0,
                                             float lo = -1.0f,
                                             float hi = 1.0f) {
  std::vector<float> v(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& x : v) x = dist(rng);
  return v;
}

// RAII device buffer：构造时 cudaMalloc，析构时 cudaFree，禁止拷贝
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t n) : n_(n) {
    CUDA_CHECK(cudaMalloc(&d_, n_ * sizeof(T)));
  }
  ~DeviceBuffer() { cudaFree(d_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void upload(const std::vector<T>& host) {
    CUDA_CHECK(cudaMemcpy(d_, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> host(n_);
    CUDA_CHECK(cudaMemcpy(host.data(), d_, n_ * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return host;
  }

  T* get() { return d_; }
  const T* get() const { return d_; }
  size_t size() const { return n_; }

 private:
  T* d_ = nullptr;
  size_t n_ = 0;
};

}  // namespace hopper
