// Phase 6b: TMA multicast —— 一条装载指令把 tile 分发到 cluster 内多个 CTA。
// 实验两件事:
//   A. 发射方语义（已实锤）: **一条 multicast 指令会给 mask 内每个 CTA 的
//      同偏移 mbarrier 各记一次 complete_tx**。因此:
//        - 各接收 CTA 只需对自己的 full barrier 做一次 arrive_expect_tx(每 tile 字节)
//        - 指令只能由一个 CTA 发射（leader, rank0）——全体发射 = 每 barrier 双重记账
//          -> 相位错乱 -> 流水线挂死（--all-issue 探针可复现）
//   B. 带宽收益: 全部 cluster 读同一条 L2 驻留 strip（64x16384 fp16 = 2MB），
//      unicast 每路独立 fetch vs multicast 一路 fetch 两路接收。
//      若 L2->SMEM 传送是瓶颈，multicast 的聚合接收带宽应约 2x。
//
// 注意（生产级流水线的差异）: 本实验 empty barrier 是每 CTA 私有的（消费者即刻
// 释放，无计算），leader 只等自己的 empty 就够了。真实 GEMM 中 rank1 的 smem 也会
// 被 multicast 覆写，释放同步须做成 cluster 级——全体消费者远程 arrive 同一个
//（leader 的）empty barrier，仅 leader 等待并发射。
//
// 流水线: S=4 级 8KB stage，full(count=1, expect_tx) / empty(count=128)，
// 消费即刻 arrive empty（无计算），issue 在迭代末尾（Phase 5 验证过的时序）。
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "common/bench.h"
#include "common/cluster.cuh"
#include "common/device_info.h"
#include "common/errors.h"
#include "common/mbarrier.cuh"
#include "common/tensor.h"
#include "common/timer.h"

namespace {

constexpr uint32_t kBlock = 128;
constexpr uint32_t kStages = 4;
constexpr uint32_t kTileBytes = 64 * 64 * 2;  // 8KB fp16
constexpr uint32_t kCols = 16384;             // strip 宽（256 个 64 宽 tile）
constexpr uint32_t kRows = 64;
constexpr uint32_t kIters = kCols / 64;

// kMcast: multicast 指令 + mask=0b11;  kLeaderOnly: 只有 rank0 发射（语义探针）
template <bool kMcast, bool kLeaderOnly>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(kBlock)
tma_mcast_kernel(const __grid_constant__ CUtensorMap tmap, uint32_t* errs_out) {
  extern __shared__ __align__(128) char smem_raw[];
  char* tiles = smem_raw;  // kStages x 8KB
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_raw + kStages * kTileBytes);
  uint64_t* empty = full + kStages;

  const uint32_t tid = threadIdx.x;
  const uint32_t rank = hopper::cluster_ctarank();

  if (tid == 0) {
    for (uint32_t s = 0; s < kStages; ++s) {
      hopper::mbarrier_init(&full[s], 1);
      hopper::mbarrier_init(&empty[s], kBlock);
    }
  }
  __syncthreads();

  auto issue = [&](uint32_t j) {
    if (j >= kStages) {
      hopper::mbarrier_wait_parity(&empty[j % kStages], ((j / kStages) - 1) & 1);
    }
    hopper::mbarrier_arrive_expect_tx(&full[j % kStages], kTileBytes);
    if (kMcast) {
      if (!kLeaderOnly || rank == 0) {
        hopper::tma_load_2d_mcast(tiles + (j % kStages) * kTileBytes, &tmap,
                                  int32_t(j * 64), 0, &full[j % kStages], 0x3);
      }
    } else {
      hopper::tma_load_2d(tiles + (j % kStages) * kTileBytes, &tmap, int32_t(j * 64),
                          0, &full[j % kStages]);
    }
  };

  if (tid == 0) {
    for (uint32_t j = 0; j + 1 < kStages && j < kIters; ++j) issue(j);
  }

  for (uint32_t i = 0; i < kIters; ++i) {
    hopper::mbarrier_wait_parity(&full[i % kStages], (i / kStages) & 1);
    if (i == kIters - 1) {  // 校验最后一级: 线性布局, G[y][x] = fp16((y*kCols+x)&1023)
      const uint32_t* su = reinterpret_cast<const uint32_t*>(tiles +
                                                             (i % kStages) * kTileBytes);
      uint32_t errs = 0;
      for (uint32_t u = tid; u < kTileBytes / 4; u += kBlock) {
        const uint32_t row = u / 32, col0 = (u % 32) * 2;
        const uint32_t e0 = (row * kCols + i * 64 + col0) & 1023;
        const uint32_t e1 = (row * kCols + i * 64 + col0 + 1) & 1023;
        const uint32_t want = uint32_t(__half_as_ushort(__float2half_rn(float(e0)))) |
                              uint32_t(__half_as_ushort(__float2half_rn(float(e1)))) << 16;
        errs += su[u] != want;
      }
      if (errs) atomicAdd(errs_out, errs);
    }
    hopper::mbarrier_arrive(&empty[i % kStages]);
    if (tid == 0 && i + kStages - 1 < kIters) {
      issue(i + kStages - 1);
    }
  }
}

CUtensorMap make_tmap_h(const __half* dptr, uint32_t rows, uint32_t cols) {
  alignas(64) CUtensorMap tmap{};
  const cuuint64_t gdim[2] = {cols, rows};
  const cuuint64_t gstride[1] = {cols * sizeof(__half)};
  const cuuint32_t box[2] = {64, 64};
  const cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, const_cast<__half*>(dptr), gdim,
      gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled 失败: CUresult %d\n", (int)r);
    std::exit(EXIT_FAILURE);
  }
  return tmap;
}

template <bool kMcast, bool kLeaderOnly>
void run(const CUtensorMap& tmap, hopper::DeviceBuffer<uint32_t>& errs) {
  const size_t smem = kStages * kTileBytes + 2 * kStages * sizeof(uint64_t);
  CUDA_CHECK(cudaMemset(errs.get(), 0, errs.size() * sizeof(uint32_t)));
  tma_mcast_kernel<kMcast, kLeaderOnly><<<156, kBlock, smem>>>(tmap, errs.get());
  CUDA_CHECK_LAST();
}

}  // namespace

int main(int argc, char** argv) {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  std::vector<__half> g(size_t(kRows) * kCols);
  for (size_t i = 0; i < g.size(); ++i) g[i] = __float2half_rn(float(i & 1023));
  hopper::DeviceBuffer<__half> dG(g.size());
  dG.upload(g);
  alignas(64) CUtensorMap tmap = make_tmap_h(dG.get(), kRows, kCols);
  hopper::DeviceBuffer<uint32_t> errs(1);

  // ---- 正确性 + 发射方语义探针 ----
  const bool probe_all = argc > 1 && std::string(argv[1]) == "--all-issue";
  run<false, false>(tmap, errs);
  std::printf("\n[CHECK ] unicast(各自装载)        : %s (mismatch=%u)\n",
              errs.download()[0] == 0 ? "PASS" : "FAIL", errs.download()[0]);
  run<true, true>(tmap, errs);
  std::printf("[CHECK ] multicast 仅rank0发射    : %s (mismatch=%u)\n",
              errs.download()[0] == 0 ? "PASS" : "FAIL", errs.download()[0]);
  if (probe_all) {
    // 反面探针（预期挂死）: 全体发射 -> 双重记账 -> 相位错乱
    run<true, false>(tmap, errs);
    std::printf("[CHECK ] multicast 全体CTA发射     : %s (mismatch=%u)\n",
                errs.download()[0] == 0 ? "PASS" : "FAIL", errs.download()[0]);
  }

  // ---- 带宽: 聚合接收 vs 实际 fetch ----
  constexpr uint32_t kGrid = 156;  // 78 cluster x 2 CTA
  const double bytes_per_cta = double(kIters) * kTileBytes;           // 2MB
  const double recv_total = double(kGrid) * bytes_per_cta;            // 接收口径
  const double fetch_mcast = double(kGrid / 2) * bytes_per_cta;       // fetch 口径(mcast)
  const double fetch_unicast = recv_total;

  std::printf("\n[BENCH ] TMA 装载, %u CTA 读同一条 2MB L2 驻留 strip\n", kGrid);
  std::printf("  %-10s %10s %14s %14s\n", "mode", "min_ms", "recv(GB/s)", "fetch(GB/s)");
  std::filesystem::create_directories("results");
  hopper::CsvWriter csv("results/tma_multicast.csv",
                        {"mode", "min_ms", "recv_gbps", "fetch_gbps"});
  {
    auto st = hopper::time_reps(3, 10, [&] { run<false, false>(tmap, errs); });
    std::printf("  %-10s %10.3f %14.1f %14.1f\n", "unicast", st.min_ms,
                recv_total / (st.min_ms * 1e-3) / 1e9, fetch_unicast / (st.min_ms * 1e-3) / 1e9);
    csv.row({"unicast", std::to_string(st.min_ms),
             std::to_string(recv_total / (st.min_ms * 1e-3) / 1e9),
             std::to_string(fetch_unicast / (st.min_ms * 1e-3) / 1e9)});
  }
  {
    auto st = hopper::time_reps(3, 10, [&] { run<true, true>(tmap, errs); });
    std::printf("  %-10s %10.3f %14.1f %14.1f\n", "mcast", st.min_ms,
                recv_total / (st.min_ms * 1e-3) / 1e9, fetch_mcast / (st.min_ms * 1e-3) / 1e9);
    csv.row({"mcast", std::to_string(st.min_ms),
             std::to_string(recv_total / (st.min_ms * 1e-3) / 1e9),
             std::to_string(fetch_mcast / (st.min_ms * 1e-3) / 1e9)});
  }
  return 0;
}
