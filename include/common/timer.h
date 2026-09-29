#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <vector>
#include "common/errors.h"

namespace hopper {

struct TimingStats {
  int reps = 0;
  double min_ms = 0.0;
  double mean_ms = 0.0;
  double max_ms = 0.0;
};

class GpuTimer {
 public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void start() { CUDA_CHECK(cudaEventRecord(start_)); }
  void stop() { CUDA_CHECK(cudaEventRecord(stop_)); }

  // 返回 start/stop 之间的毫秒数（阻塞至 stop 事件完成）
  double elapsed_ms() {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(stop_));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

// 先空跑 warmup 次，再正式计时 reps 次（每次独立 sync，取 min/mean/max）
inline TimingStats time_reps(int warmup, int reps,
                             const std::function<void()>& launch) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> samples(reps);
  GpuTimer timer;
  for (int i = 0; i < reps; ++i) {
    timer.start();
    launch();
    timer.stop();
    samples[i] = timer.elapsed_ms();
  }

  TimingStats st;
  st.reps = reps;
  double sum = 0.0;
  st.min_ms = samples[0];
  st.max_ms = samples[0];
  for (double s : samples) {
    sum += s;
    if (s < st.min_ms) st.min_ms = s;
    if (s > st.max_ms) st.max_ms = s;
  }
  st.mean_ms = sum / reps;
  return st;
}

}  // namespace hopper
