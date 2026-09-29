// Phase 1b: TMA 2D tensor map + swizzle。
// 学两件事:
//   1. cuTensorMapEncodeTiled / cp.async.bulk.tensor.2d 的真实用法（GEMM tile 搬运的标准形态）
//   2. SWIZZLE_128B 的 smem 地址变换公式: 行内 16B chunk 索引 ^= (row % 8)
//      —— 这是后续 wgmma 读 shared memory 操作数的必备前置知识。
// 消费方式: 每线程读固定列的一个元素(256线程 = 128行 x 2列)。
// NONE 下同一 warp 32 线程全打同一 bank(32-way 冲突), 128B swizzle 散开为 4-way。
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <vector>

#include "common/bench.h"
#include "common/checker.h"
#include "common/device_info.h"
#include "common/tensor.h"
#include "common/timer.h"

namespace {

constexpr uint32_t kCols = 8192;    // 矩阵列数（内维，globalDim[0]）
constexpr uint32_t kRows = 8192;    // 矩阵行数（globalDim[1]）
constexpr uint32_t kBoxCols = 32;   // box 内维 32 floats = 128B（= swizzle 宽度）
constexpr uint32_t kBoxRows = 128;  // box 外维
constexpr uint32_t kBoxElems = kBoxCols * kBoxRows;  // 4096 floats
constexpr uint32_t kBoxBytes = kBoxElems * 4;        // 16KB
constexpr uint32_t kColBoxes = kCols / kBoxCols;     // 256
constexpr uint32_t kRowBoxes = kRows / kBoxRows;     // 64
constexpr uint32_t kBoxes = kColBoxes * kRowBoxes;   // 16384
constexpr uint32_t kBlock = 256;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <bool kSwizzle128>
__global__ void __launch_bounds__(kBlock)
tma2d_kernel(const __grid_constant__ CUtensorMap tmap,
             float* __restrict__ out) {
  __shared__ alignas(128) float tile[kBoxElems];  // 16KB, 128B 对齐(128B swizzle 要求)
  __shared__ alignas(8) uint64_t bar;
  const uint32_t tid = threadIdx.x;

  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&bar)));
  }
  __syncthreads();

  uint32_t i = 0;  // 本 block 第 i 个 box <-> mbarrier phase 奇偶
  for (uint32_t box = blockIdx.x; box < kBoxes; box += gridDim.x, ++i) {
    const uint32_t bx = box % kColBoxes;
    const uint32_t by = box / kColBoxes;
    if (tid == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                       smem_u32(&bar)),
                   "r"(kBoxBytes));
      // 坐标是元素单位: {x=dim0 起始列, y=dim1 起始行}
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
          ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
              smem_u32(tile)),
          "l"(&tmap), "r"(bx * kBoxCols), "r"(by * kBoxRows), "r"(
              smem_u32(&bar)));
    }
    asm volatile(
        "{\n\t.reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t}\n" ::"r"(smem_u32(&bar)),
        "r"(i & 1));

    // 消费: 线程 t 读 (row = t%128, col = 4 + t/128)，即每 warp 32 行同列
    const uint32_t row = tid & (kBoxRows - 1);
    const uint32_t col = 4 + (tid >> 7);
    uint32_t word;
    if constexpr (kSwizzle128) {
      // SWIZZLE_128B: 16B chunk 号 (col/4) 异或 (row%8)，chunk 内偏移不变
      word = row * kBoxCols + (((col >> 2) ^ (row & 7)) << 2) + (col & 3);
    } else {
      word = row * kBoxCols + col;
    }
    out[(size_t)box * kBlock + tid] = tile[word];
  }
}

CUtensorMap make_tmap(const float* dptr, CUtensorMapSwizzle sw) {
  alignas(64) CUtensorMap tmap{};
  const cuuint64_t gdim[2] = {kCols, kRows};
  const cuuint64_t gstride[1] = {kCols * sizeof(float)};  // dim1 步长(字节)
  const cuuint32_t box[2] = {kBoxCols, kBoxRows};
  const cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<float*>(dptr), gdim,
      gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled 失败 (swizzle=%d): CUresult %d\n",
                 (int)sw, (int)r);
    std::exit(EXIT_FAILURE);
  }
  return tmap;
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  const size_t n = size_t(kCols) * kRows;
  auto hin = hopper::make_random_vector(n, 11);
  hopper::DeviceBuffer<float> din(n);
  din.upload(hin);

  // 期望输出: out[box*256 + t] = 矩阵在 (全局行, 全局列) 的值
  std::vector<float> expected((size_t)kBoxes * kBlock);
  for (uint32_t box = 0; box < kBoxes; ++box) {
    const uint32_t bx = box % kColBoxes, by = box / kColBoxes;
    for (uint32_t t = 0; t < kBlock; ++t) {
      const uint32_t row = t & (kBoxRows - 1), col = 4 + (t >> 7);
      expected[(size_t)box * kBlock + t] =
          hin[(size_t)(by * kBoxRows + row) * kCols + bx * kBoxCols + col];
    }
  }
  hopper::DeviceBuffer<float> dout((size_t)kBoxes * kBlock);

  struct Mode {
    const char* name;
    CUtensorMapSwizzle sw;
    bool read_swizzled;
  };
  std::vector<Mode> modes = {
      {"swizzle_none", CU_TENSOR_MAP_SWIZZLE_NONE, false},
      {"swizzle_128B", CU_TENSOR_MAP_SWIZZLE_128B, true},
  };

  const double bytes_per_pass = double(kBoxes) * kBoxBytes;  // 只计读
  const double peak = hopper::theoretical_bw_gbps(caps);

  std::printf(
      "\n[BENCH ] TMA 2D: %ux%u fp32 矩阵, box=%ux%u (16KB), 消费=列读(bank conflict 敏感)\n",
      kCols, kRows, kBoxCols, kBoxRows);
  std::printf("         数据搬运量/pass = %.0f MB (只计读)\n\n",
              bytes_per_pass / (1024.0 * 1024.0));
  std::printf("  %-12s %6s %10s %12s %10s %8s\n", "mode", "grid", "blk/SM",
              "min_ms", "GB/s", "%peak");

  struct Row {
    const char* mode;
    int grid, bpsm;
    hopper::TimingStats st;
    double bw;
  };
  std::vector<Row> rows;

  // ---- 正确性（先全部过一遍）----
  for (auto& m : modes) {
    alignas(64) CUtensorMap tmap = make_tmap(din.get(), m.sw);
    if (m.read_swizzled) {
      tma2d_kernel<true><<<caps.sm_count * 2, kBlock>>>(tmap, dout.get());
    } else {
      tma2d_kernel<false><<<caps.sm_count * 2, kBlock>>>(tmap, dout.get());
    }
    CUDA_CHECK_LAST();
    auto rep = hopper::check_close(dout.download(), expected, 0.0, 0.0);
    hopper::print_report(rep, m.name);
    if (!rep.pass) return 1;
  }

  // ---- 基准: grid 扫描 ----
  for (auto& m : modes) {
    alignas(64) CUtensorMap tmap = make_tmap(din.get(), m.sw);
    auto launch = [&](int grid) {
      if (m.read_swizzled) {
        tma2d_kernel<true><<<grid, kBlock>>>(tmap, dout.get());
      } else {
        tma2d_kernel<false><<<grid, kBlock>>>(tmap, dout.get());
      }
      CUDA_CHECK_LAST();
    };
    for (int bpsm : {1, 2, 4, 8}) {
      const int grid = caps.sm_count * bpsm;
      auto st = hopper::time_reps(5, 30, [&] { launch(grid); });
      double bw = hopper::bandwidth_gbps(bytes_per_pass, st.min_ms);
      std::printf("  %-12s %6d %10d %12.4f %12.1f %7.1f%%\n", m.name, grid,
                  bpsm, st.min_ms, bw, bw / peak * 100.0);
      rows.push_back({m.name, grid, bpsm, st, bw});
    }
  }

  std::filesystem::create_directories("results");
  hopper::CsvWriter csv("results/tma2d_swizzle.csv",
                        {"mode", "grid", "blocks_per_sm", "min_ms", "mean_ms",
                         "gbps", "pct_peak"});
  for (auto& r : rows) {
    csv.row({r.mode, std::to_string(r.grid), std::to_string(r.bpsm),
             std::to_string(r.st.min_ms), std::to_string(r.st.mean_ms),
             std::to_string(r.bw), std::to_string(r.bw / peak * 100.0)});
  }
  return 0;
}
