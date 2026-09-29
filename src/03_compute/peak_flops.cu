// Phase 3 (算力密度基线): 纯寄存器循环测 CUDA core / tensor core 峰值。
// 不碰内存，只看发射吞吐——这是后续所有 TFLOPS 数据的分母。
// H20 是"砍算力保带宽"的 Hopper，张量核峰值与 H100 差异巨大，必须实测。
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "common/bench.h"
#include "common/device_info.h"
#include "common/errors.h"
#include "common/tensor.h"
#include "common/timer.h"

namespace {

constexpr uint32_t kBlock = 256;

// FP32 FFMA: 8 条独立累加链 (ILP=8)，无内存访问
__global__ void __launch_bounds__(kBlock)
ffma_peak_kernel(float* __restrict__ out, uint32_t iters) {
  const float b = 1.0000001f, c = 1.1e-7f;
  float a0 = threadIdx.x * 1e-9f + 0.1f;
  float a1 = a0 + 1e-3f, a2 = a0 + 2e-3f, a3 = a0 + 3e-3f;
  float a4 = a0 + 4e-3f, a5 = a0 + 5e-3f, a6 = a0 + 6e-3f, a7 = a0 + 7e-3f;
  for (uint32_t i = 0; i < iters; ++i) {
    a0 = fmaf(b, c, a0); a1 = fmaf(b, c, a1); a2 = fmaf(b, c, a2); a3 = fmaf(b, c, a3);
    a4 = fmaf(b, c, a4); a5 = fmaf(b, c, a5); a6 = fmaf(b, c, a6); a7 = fmaf(b, c, a7);
  }
  out[blockIdx.x * kBlock + threadIdx.x] =
      ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}

// FP16 tensor core: mma.sync m16n8k16 (4096 flop/条/warp)，寄存器内反复发射。
// 每条 mma 4096 flop；Hopper 张量核对 mma.sync 与 wgmma 吞吐一致（发射粒度不同）。
__global__ void __launch_bounds__(kBlock)
mma_f16_peak_kernel(float* __restrict__ out, uint32_t iters) {
  // fragment 值任意（只测吞吐，不验结果）
  const unsigned a0 = threadIdx.x * 3u + 0u, a1 = threadIdx.x * 3u + 1u,
                 a2 = threadIdx.x * 3u + 2u, a3 = threadIdx.x * 5u + 1u;
  const unsigned b0 = threadIdx.x * 7u + 2u, b1 = threadIdx.x * 11u + 3u;
  float d0 = 0.f, d1 = 1.f, d2 = 2.f, d3 = 3.f;
  for (uint32_t i = 0; i < iters; ++i) {
#pragma unroll 8
    for (int u = 0; u < 8; ++u) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  out[blockIdx.x * kBlock + threadIdx.x] = d0 + d1 + d2 + d3;
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  const int grid = caps.sm_count * 8;  // 2048 线程/SM，满占用
  hopper::DeviceBuffer<float> out(size_t(grid) * kBlock);

  struct Result {
    const char* name;
    double tflops;
    double ref_tflops;  // 理论/架构推算参照
  };
  std::vector<Result> results;

  // ---- FP32 FFMA ----
  {
    const uint32_t iters = 200000;
    const double flops = 2.0 * 8.0 * iters * double(grid) * kBlock;
    auto st = hopper::time_reps(3, 10, [&] {
      ffma_peak_kernel<<<grid, kBlock>>>(out.get(), iters);
      CUDA_CHECK_LAST();
    });
    // 架构推算: 78 SM x 128 FMA lane x 2 flop x 1.98 GHz（clock_khz/1e3 = MHz，/1e6 = TFLOPS）
    const double ref = 2.0 * caps.sm_count * 128.0 * (caps.clock_khz / 1000.0) / 1e6;
    results.push_back({"FP32 FFMA (CUDA core)", hopper::tflops(flops, st.min_ms), ref});
  }

  // ---- FP16 tensor (mma.sync) ----
  {
    const uint32_t iters = 20000;
    const double warps = double(grid) * (kBlock / 32);
    const double flops = 4096.0 * 8.0 * iters * warps;  // 2*16*8*16 per mma
    auto st = hopper::time_reps(3, 10, [&] {
      mma_f16_peak_kernel<<<grid, kBlock>>>(out.get(), iters);
      CUDA_CHECK_LAST();
    });
    results.push_back({"FP16 tensor (mma.sync)", hopper::tflops(flops, st.min_ms), 0.0});
  }

  // ---- 输出 ----
  std::printf("\n[PEAK  ] 算力密度实测（纯寄存器循环，无内存流量；满载 GPU 下偏低）\n");
  for (auto& r : results) {
    if (r.ref_tflops > 0.0) {
      std::printf("  %-24s : %8.2f TFLOPS  (架构推算 %.2f, 达成 %.0f%%)\n", r.name,
                  r.tflops, r.ref_tflops, r.tflops / r.ref_tflops * 100.0);
    } else {
      std::printf("  %-24s : %8.2f TFLOPS  (为 FP32 FFMA 的 %.1f 倍)\n", r.name,
                  r.tflops, r.tflops / results[0].tflops);
    }
  }

  std::filesystem::create_directories("results");
  hopper::CsvWriter csv("results/peak_flops.csv", {"kernel", "tflops"});
  for (auto& r : results) csv.row({r.name, std::to_string(r.tflops)});
  return 0;
}
