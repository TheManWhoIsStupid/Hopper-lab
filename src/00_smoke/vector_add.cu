// Phase 0 冒烟测试: vector add。
// 目的不是算子本身，而是打通整个框架链路:
//   设备查询 -> 数据准备 -> kernel -> 正确性检查 -> 性能基准 -> CSV 落盘
#include <cstdio>
#include <filesystem>
#include <vector>

#include "common/bench.h"
#include "common/checker.h"
#include "common/device_info.h"
#include "common/tensor.h"
#include "common/timer.h"

__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  constexpr size_t kN = 1ull << 25;  // 32M elements, 3 x 128MB 显存占用
  auto ha = hopper::make_random_vector(kN, 42);
  auto hb = hopper::make_random_vector(kN, 43);
  std::vector<float> href(kN);
  for (size_t i = 0; i < kN; ++i) href[i] = ha[i] + hb[i];

  hopper::DeviceBuffer<float> da(kN), db(kN), dc(kN);
  da.upload(ha);
  db.upload(hb);

  const size_t bytes_moved = 3 * kN * sizeof(float);  // 2 读 + 1 写
  constexpr size_t kBlock = 256;

  auto launch = [&] {
    vector_add_kernel<<<(kN + kBlock - 1) / kBlock, kBlock>>>(
        da.get(), db.get(), dc.get(), kN);
    CUDA_CHECK_LAST();
  };

  // ---- 正确性 ----
  launch();
  auto hgot = dc.download();
  auto report = hopper::check_close(hgot, href);
  hopper::print_report(report, "vector_add");

  // ---- 性能 ----
  auto stats = hopper::time_reps(20, 100, launch);
  double peak = hopper::theoretical_bw_gbps(caps);
  double bw = hopper::bandwidth_gbps(bytes_moved, stats.min_ms);
  std::printf(
      "[BENCH ] vector_add : n=%zu  min=%.4f ms  mean=%.4f ms  BW=%.1f GB/s "
      "(%.1f%% of ~%.0f GB/s peak)\n",
      kN, stats.min_ms, stats.mean_ms, bw, bw / peak * 100.0, peak);
  std::printf("  (注: GPU 被其他任务占用时数据偏噪声, 以 min_ms 为准)\n");

  // ---- 落盘 ----
  std::filesystem::create_directories("results");
  hopper::CsvWriter csv("results/smoke_vector_add.csv",
                        {"kernel", "n", "min_ms", "mean_ms", "gbps",
                         "pct_peak"});
  csv.row({"vector_add", std::to_string(kN), std::to_string(stats.min_ms),
           std::to_string(stats.mean_ms), std::to_string(bw),
           std::to_string(bw / peak * 100.0)});

  return report.pass ? 0 : 1;
}
