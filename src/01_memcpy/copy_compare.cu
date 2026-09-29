// Phase 1: global -> shared 搬运机制对比 (naive / cp.async / TMA bulk)。
// 正确性: 每机制跑一遍，与 host 期望值精确比对（纯搬运应 bit-exact）。
// 性能:   grid 大小扫描 (1/2/4/7 blocks per SM)，验证"少线程打满带宽"的 Hopper 卖点。
#include <cstdio>
#include <filesystem>
#include <functional>
#include <utility>
#include <vector>

#include "common/bench.h"
#include "common/checker.h"
#include "common/device_info.h"
#include "common/tensor.h"
#include "common/timer.h"
#include "copy_kernels.cuh"

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  constexpr uint32_t kTiles = 8192;  // 8192 * 32KB = 256MB
  const size_t n_floats = size_t(kTiles) * hopper::kCopyTileElems;

  auto hin = hopper::make_random_vector(n_floats, 7);
  std::vector<float> expected(kTiles);
  for (uint32_t t = 0; t < kTiles; ++t) {
    uint32_t rot = (t * 13) & (hopper::kCopyTileElems - 1);
    expected[t] = hin[size_t(t) * hopper::kCopyTileElems + rot];
  }

  hopper::DeviceBuffer<float> din(n_floats);
  hopper::DeviceBuffer<float> dout(kTiles);
  din.upload(hin);

  using Launch = std::function<void(int)>;
  std::vector<std::pair<const char*, Launch>> mechs = {
      {"naive",
       [&](int grid) {
         hopper::copy_naive_kernel<<<grid, hopper::kCopyBlock>>>(
             reinterpret_cast<const float4*>(din.get()), dout.get(), kTiles);
         CUDA_CHECK_LAST();
       }},
      {"cp.async",
       [&](int grid) {
         hopper::copy_cpasync_kernel<<<grid, hopper::kCopyBlock>>>(
             reinterpret_cast<const float4*>(din.get()), dout.get(), kTiles);
         CUDA_CHECK_LAST();
       }},
      {"tma1d",
       [&](int grid) {
         hopper::copy_tma1d_kernel<<<grid, hopper::kCopyBlock>>>(
             din.get(), dout.get(), kTiles);
         CUDA_CHECK_LAST();
       }},
  };

  // ---- 正确性（纯搬运，期望 bit-exact）----
  for (auto& mech : mechs) {
    const char* name = mech.first;
    auto& launch = mech.second;
    launch(caps.sm_count * 2);
    auto rep = hopper::check_close(dout.download(), expected, 0.0, 0.0);
    hopper::print_report(rep, name);
    if (!rep.pass) return 1;
  }

  // ---- 基准: grid 扫描 ----
  const double bytes_per_pass = double(size_t(kTiles) * hopper::kCopyTileBytes);
  const double peak = hopper::theoretical_bw_gbps(caps);
  std::printf("\n[BENCH ] 256MB global->shared, tile=32KB, block=%u threads\n",
              hopper::kCopyBlock);
  std::printf("         数据搬运量/pass = %.0f MB, GPU 空闲时数据更有代表性\n\n",
              bytes_per_pass / (1024.0 * 1024.0));
  std::printf("  %-10s %6s %10s %12s %10s %8s\n", "mech", "grid", "blk/SM",
              "min_ms", "GB/s", "%peak");

  std::filesystem::create_directories("results");
  hopper::CsvWriter csv("results/copy_compare.csv",
                        {"mech", "grid", "blocks_per_sm", "min_ms", "mean_ms",
                         "gbps", "pct_peak"});

  for (int bpsm : {1, 2, 4, 7}) {
    const int grid = caps.sm_count * bpsm;
    for (auto& mech : mechs) {
      const char* name = mech.first;
      auto& launch = mech.second;
      auto st = hopper::time_reps(5, 30, [&] { launch(grid); });
      double bw = hopper::bandwidth_gbps(bytes_per_pass, st.min_ms);
      std::printf("  %-10s %6d %10d %12.4f %12.1f %7.1f%%\n", name, grid, bpsm,
                  st.min_ms, bw, bw / peak * 100.0);
      csv.row({name, std::to_string(grid), std::to_string(bpsm),
               std::to_string(st.min_ms), std::to_string(st.mean_ms),
               std::to_string(bw), std::to_string(bw / peak * 100.0)});
    }
  }
  return 0;
}
