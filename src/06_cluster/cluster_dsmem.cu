// Phase 6a: 2-CTA cluster + DSMEM 机制验证。
//   1) 远程读：写本地 smem -> cluster.sync -> 读邻居 smem，校验
//   2) 远程写：st.shared::cluster 直接写邻居 smem，邻居校验
//   3) 远程 mbarrier arrive：邻居来 arrive 我的 barrier，我等 phase 翻转
//   4) 带宽：本地 ld.shared.v4 vs 远程 ld.shared::cluster.v4（uint4 连读）
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "common/bench.h"
#include "common/cluster.cuh"
#include "common/device_info.h"
#include "common/errors.h"
#include "common/gmma.cuh"
#include "common/mbarrier.cuh"
#include "common/tensor.h"
#include "common/timer.h"

namespace {

constexpr uint32_t kBlock = 128;
constexpr uint32_t kN = 4096;    // uint32 元素数 = 16KB / CTA
constexpr uint32_t kRounds = 2000;

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(kBlock)
cluster_dsmem_kernel(uint32_t* out,       // [block][4]: 读错/写错/bar_ok/防死码
                     uint64_t* cycles) {  // [block][2]: 本地/远程读耗时(cycle)
  __shared__ __align__(16) uint32_t sdata[kN];
  __shared__ uint64_t bar;

  const uint32_t tid = threadIdx.x;
  const uint32_t rank = hopper::cluster_ctarank();
  const uint32_t nb = rank ^ 1u;  // 2-CTA cluster 的邻居
  const uint32_t bid = blockIdx.x;

  // ---- 1) 远程读 ----
  for (uint32_t i = tid; i < kN; i += kBlock) sdata[i] = rank * 1000000 + i;
  hopper::cluster_sync();
  const uint32_t remote = hopper::mapa(hopper::smem_u32(sdata), nb);
  uint32_t errs = 0;
  for (uint32_t i = tid; i < kN; i += kBlock)
    errs += hopper::ld_shared_cluster_u32(remote + i * 4) != nb * 1000000 + i;

  // ---- 2) 远程写 ----
  for (uint32_t i = tid; i < kN; i += kBlock)
    hopper::st_shared_cluster_u32(remote + i * 4, rank * 2000000 + i);
  hopper::cluster_sync();
  for (uint32_t i = tid; i < kN; i += kBlock)
    errs += sdata[i] != nb * 2000000 + i;

  // ---- 3) 远程 mbarrier arrive ----
  // 注意 barrier.cluster 的 .aligned 语义: 必须全 CTA 收敛执行，不能埋在 tid 分支里
  //（warp 0 内 lane0 与其余 lane 分歧执行 cluster barrier = 未定义行为，实测挂死）。
  if (tid == 0) {
    hopper::mbarrier_init(&bar, 1);
    hopper::fence_mbarrier_init();
  }
  __syncthreads();
  hopper::cluster_sync();  // init 对邻居 CTA 可见
  if (tid == 0) {
    hopper::mbarrier_arrive_remote(hopper::mapa(hopper::smem_u32(&bar), nb));
  }
  hopper::mbarrier_wait_parity(&bar, 0);  // 邻居的那次 arrive 到位才通过
  uint32_t bar_ok = 1;

  // ---- 4) 带宽：本地 vs 远程 uint4 连读 ----
  for (uint32_t i = tid; i < kN; i += kBlock)
    sdata[i] = i * 2654435761u;  // 重新填非平凡数据
  hopper::cluster_sync();

  const uint32_t local = hopper::smem_u32(sdata);
  for (uint32_t p = 0; p < 2; ++p) {
    const uint32_t base = (p == 0) ? local : remote;
    uint32_t acc = 0;
    const long long t0 = clock64();
    for (uint32_t r = 0; r < kRounds; ++r) {
      for (uint32_t i = tid; i < kN / 4; i += kBlock) {
        const uint4 v = (p == 0) ? hopper::ld_shared_u128(base + i * 16)
                                 : hopper::ld_shared_cluster_u128(base + i * 16);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
      }
    }
    const long long t1 = clock64();
    cycles[bid * 2 + p] = uint64_t(t1 - t0);
    if (acc == 0xdeadbeefu) out[bid * 4 + 3] = 1;  // 消费 acc
    hopper::cluster_sync();  // 分隔两个 pass
  }

  if (tid == 0) {
    out[bid * 4 + 0] = errs;
    out[bid * 4 + 1] = 0;  // 预留（写错误并入 errs）
    out[bid * 4 + 2] = bar_ok;
  }
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  constexpr uint32_t kGrid = 156;  // 78 个 2-CTA cluster，每 SM 2 CTA
  static_assert(kGrid % 2 == 0, "grid 须被 cluster 维度整除");

  hopper::DeviceBuffer<uint32_t> out(kGrid * 4);
  hopper::DeviceBuffer<uint64_t> cyc(kGrid * 2);
  CUDA_CHECK(cudaMemset(out.get(), 0, out.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(cyc.get(), 0, cyc.size() * sizeof(uint64_t)));

  cluster_dsmem_kernel<<<kGrid, kBlock>>>(out.get(), cyc.get());
  CUDA_CHECK_LAST();

  auto h = out.download();
  auto c = cyc.download();

  uint32_t max_errs = 0, bar_bad = 0;
  for (uint32_t b = 0; b < kGrid; ++b) {
    max_errs = std::max(max_errs, h[b * 4]);
    bar_bad += (h[b * 4 + 2] == 0);
  }
  std::printf("\n[CHECK ] DSMEM 交换: %s (mismatch=%u, 远程mbarrier失败CTA=%u)\n",
              (max_errs == 0 && bar_bad == 0) ? "PASS" : "FAIL", max_errs, bar_bad);

  // 带宽统计（每 CTA 读了 kRounds*16KB）
  const double bytes = double(kRounds) * kN * 4;
  double gb_min[2] = {1e9, 1e9}, gb_sum[2] = {0, 0};
  for (uint32_t b = 0; b < kGrid; ++b) {
    for (uint32_t p = 0; p < 2; ++p) {
      const double sec = double(c[b * 2 + p]) / 1.98e9;  // SM 时钟 1.98GHz
      const double gb = bytes / sec / 1e9;
      gb_min[p] = std::min(gb_min[p], gb);
      gb_sum[p] += gb;
    }
  }
  const char* names[2] = {"本地 ld.shared.v4      ", "远程 ld.shared::cluster"};
  std::printf("\n[BENCH ] smem 读带宽 (uint4, %u CTA, 每 CTA %.0f MB)\n", kGrid,
              bytes / 1e6);
  std::printf("  %s  min(GB/s)  mean(GB/s)\n", "模式                    ");
  std::filesystem::create_directories("results");
  hopper::CsvWriter csv("results/cluster_dsmem.csv", {"mode", "min_gbps", "mean_gbps"});
  for (uint32_t p = 0; p < 2; ++p) {
    std::printf("  %s  %10.1f  %11.1f\n", names[p], gb_min[p], gb_sum[p] / kGrid);
    csv.row({p == 0 ? "local" : "remote", std::to_string(gb_min[p]),
             std::to_string(gb_sum[p] / kGrid)});
  }
  return (max_errs == 0 && bar_bad == 0) ? 0 : 1;
}
