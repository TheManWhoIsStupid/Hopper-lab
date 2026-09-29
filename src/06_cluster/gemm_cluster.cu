// Phase 6c: multicast GEMM —— Phase 5 的 TMA+wgmma 流水线加上 cluster 数据复用。
//
//   grid dim3(N/64, M/64)，__cluster_dims__(2,1,1)：blockIdx.x 相邻两 CTA 成对，
//   同 bm 不同 bn —— 共享 A tile。A 走 multicast（leader/rank0 发射一条，两 CTA
//   各收一份），B 各自 unicast。fetch 侧 A 流量减半。
//
// 屏障拓扑（6b 语义实验的直接应用）:
//   full[s]    每 CTA 私有, count=1: 各自 tid0 登记本 stage 的 16KB expect
//   empty_a[s] 仅 leader, count=256: 两 CTA 全体消费者 arrive（rank1 走远程
//              mbarrier.arrive.shared::cluster）——multicast 覆写两边的 A 区，
//              释放必须双 CTA 确认；try_wait 只能本地等，所以放 leader
//   empty_b[s] 每 CTA 私有, count=128: 自己的 B 区自己管
//   armed[s]   仅 leader, count=1: rank1 登记 expect 之后远程 arrive，
//              rank0 等 armed 再发射 multicast —— 堵住"记账先于登记落地"的
//              跨 CTA 竞态窗口（tx 记账是相位敏感的）
//
// 其余（wgmma 时序、issue 在迭代末尾、wait_group<1> 滞后释放）与 Phase 5 相同。
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "common/bench.h"
#include "common/checker.h"
#include "common/cluster.cuh"
#include "common/device_info.h"
#include "common/errors.h"
#include "common/gmma.cuh"
#include "common/mbarrier.cuh"
#include "common/tensor.h"
#include "common/timer.h"

namespace {

constexpr uint32_t kBM = 64, kBN = 64, kBK = 64;
constexpr uint32_t kTileBytes = kBM * kBK * 2;   // 8KB (fp16)
constexpr uint32_t kStageBytes = 2 * kTileBytes; // A + B = 16KB
constexpr uint32_t kBlock = 128;                 // 一个 warpgroup

template <uint32_t S>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(kBlock)
gemm_cluster_kernel(const __grid_constant__ CUtensorMap tmap_a,
                    const __grid_constant__ CUtensorMap tmap_b,
                    float* __restrict__ D, uint32_t M, uint32_t N, uint32_t K) {
  extern __shared__ __align__(1024) char smem_raw[];
  char* sA = smem_raw;                  // S x 8KB, SW128 布局（TMA 写 = wgmma 读）
  char* sB = smem_raw + S * kTileBytes; // S x 8KB
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_raw + 2 * S * kTileBytes);
  uint64_t* empty_a = full + S;   // 仅 leader 的实例被使用
  uint64_t* empty_b = empty_a + S;
  uint64_t* armed = empty_b + S;  // 仅 leader 的实例被使用

  const uint32_t tid = threadIdx.x;
  const uint32_t rank = hopper::cluster_ctarank();  // = blockIdx.x % 2
  const uint32_t bm = blockIdx.y, bn = blockIdx.x;
  const uint32_t kblocks = K / kBK;

  if (tid == 0) {
    for (uint32_t s = 0; s < S; ++s) {
      hopper::mbarrier_init(&full[s], 1);
      hopper::mbarrier_init(&empty_b[s], kBlock);
      if (rank == 0) {
        hopper::mbarrier_init(&empty_a[s], 2 * kBlock);
        hopper::mbarrier_init(&armed[s], 1);
      }
    }
    hopper::fence_mbarrier_init();  // 远程 arrive 前必须（6a 踩坑）
  }
  __syncthreads();
  hopper::cluster_sync();  // barrier init 对整个 cluster 可见；须全体收敛执行

  auto issue = [&](uint32_t j) {
    const uint32_t s = j % S;
    if (j >= S) {
      hopper::mbarrier_wait_parity(&empty_b[s], ((j / S) - 1) & 1);
      if (rank == 0) {
        hopper::mbarrier_wait_parity(&empty_a[s], ((j / S) - 1) & 1);
      }
    }
    hopper::mbarrier_arrive_expect_tx(&full[s], kStageBytes);
    if (rank == 0) {
      hopper::mbarrier_wait_parity(&armed[s], (j / S) & 1);
      hopper::tma_load_2d_mcast(sA + s * kTileBytes, &tmap_a, int32_t(j * kBK),
                                int32_t(bm * kBM), &full[s], 0x3);
      hopper::tma_load_2d(sB + s * kTileBytes, &tmap_b, int32_t(j * kBK),
                          int32_t(bn * kBN), &full[s]);
    } else {
      // 先登记 expect 再向 leader 报 armed，rank0 见到 armed 才发射 multicast
      hopper::mbarrier_arrive_remote(hopper::mapa(hopper::smem_u32(&armed[s]), 0));
      hopper::tma_load_2d(sB + s * kTileBytes, &tmap_b, int32_t(j * kBK),
                          int32_t(bn * kBN), &full[s]);
    }
  };

  if (tid == 0) {
    for (uint32_t j = 0; j + 1 < S && j < kblocks; ++j) issue(j);
  }

  float d[32] = {};
  hopper::wgmma_fence();
  for (uint32_t i = 0; i < kblocks; ++i) {
    hopper::mbarrier_wait_parity(&full[i % S], (i / S) & 1);
    const char* sa = sA + (i % S) * kTileBytes;
    const char* sb = sB + (i % S) * kTileBytes;
#pragma unroll
    for (uint32_t kk = 0; kk < kBK / 16; ++kk) {
      const uint64_t da = hopper::gmma_desc_k_sw128(sa + kk * 32, 1024, 0);
      const uint64_t db = hopper::gmma_desc_k_sw128(sb + kk * 32, 1024, 0);
      hopper::wgmma_m64n64k16_f32_f16(da, db, d, (i || kk) ? 1u : 0u);
    }
    hopper::wgmma_commit_group();
    if (i >= 1) {
      hopper::wgmma_wait_group<1>();
      const uint32_t s = (i - 1) % S;
      // A 区释放须让 leader 知道（rank1 远程 arrive empty_a）；B 区各自本地
      hopper::mbarrier_arrive_remote(hopper::mapa(hopper::smem_u32(&empty_a[s]), 0));
      hopper::mbarrier_arrive(&empty_b[s]);
    }
    if (tid == 0 && i + S - 1 < kblocks) {
      issue(i + S - 1);
    }
  }
  hopper::wgmma_wait_group<0>();

  // 收尾同步：最后一轮的远程 arrive（rank1 -> rank0 的 empty_a/armed）可能还在
  // 互连上。没有这道 cluster barrier 的话，先退出的 CTA smem 被回收，迟到的
  // 远程 arrive 打到已释放的地址 = ULF（8192^3 S=3 实测复现）。
  // barrier.cluster 的 release 语义保证先于此的远程访存已落地。
  hopper::cluster_sync();

  const uint32_t warp = tid / 32, lane = tid % 32;
#pragma unroll
  for (uint32_t r = 0; r < 32; ++r) {
    const uint32_t m = 16 * warp + lane / 4 + 8 * ((r / 2) % 2);
    const uint32_t n = 2 * (lane % 4) + r % 2 + 8 * (r / 4);
    D[size_t(bm * kBM + m) * N + bn * kBN + n] = d[r];
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
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled 失败: CUresult %d\n", (int)r);
    std::exit(EXIT_FAILURE);
  }
  return tmap;
}

template <uint32_t S>
void launch_config(const CUtensorMap& ta, const CUtensorMap& tb, float* d,
                   uint32_t M, uint32_t N, uint32_t K) {
  const size_t smem = 2 * S * kTileBytes + 4 * S * sizeof(uint64_t);
  CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(gemm_cluster_kernel<S>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  gemm_cluster_kernel<S><<<dim3(N / kBN, M / kBM), kBlock, smem>>>(ta, tb, d, M, N,
                                                                  K);
  CUDA_CHECK_LAST();
}

void dispatch(uint32_t stages, const CUtensorMap& ta, const CUtensorMap& tb,
              float* d, uint32_t M, uint32_t N, uint32_t K) {
  switch (stages) {
    case 2: launch_config<2>(ta, tb, d, M, N, K); break;
    case 3: launch_config<3>(ta, tb, d, M, N, K); break;
    case 4: launch_config<4>(ta, tb, d, M, N, K); break;
    case 6: launch_config<6>(ta, tb, d, M, N, K); break;
    default: std::fprintf(stderr, "未支持的 stage 数: %u\n", stages); std::exit(1);
  }
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  // ---- 正确性: 512x512x256, S in {2,3,4}, 每个 S 重复 5 次抖竞态 ----
  {
    constexpr uint32_t M = 512, N = 512, K = 256;
    auto ha = hopper::make_random_vector(size_t(M) * K, 501, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 502, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_h(dA.get(), M, K);
    alignas(64) CUtensorMap tb = make_tmap_h(dB.get(), N, K);

    std::vector<float> ref(size_t(M) * N);
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t n = 0; n < N; ++n) {
        float acc = 0.f;
        for (uint32_t k = 0; k < K; ++k)
          acc += float(a[size_t(m) * K + k]) * float(b[size_t(n) * K + k]);
        ref[size_t(m) * N + n] = acc;
      }

    for (uint32_t s : {2u, 3u, 4u}) {
      bool ok = true;
      for (uint32_t rep = 0; rep < 5; ++rep) {
        dispatch(s, ta, tb, dC.get(), M, N, K);
        auto r = hopper::check_close(dC.download(), ref, 1e-3, 1e-3);
        ok = ok && r.pass;
        if (!r.pass) {
          char tag[48];
          std::snprintf(tag, sizeof(tag), "cluster S=%u rep=%u", s, rep);
          hopper::print_report(r, tag);
        }
      }
      std::printf("[CHECK ] cluster S=%u x5           : %s\n", s,
                  ok ? "PASS" : "FAIL");
      if (!ok) return 1;
    }
  }

  // ---- 吞吐: 2048 与 8192（对照 Phase 5 无 multicast: 103.6T / 72.2T）----
  std::filesystem::create_directories("results");
  {
    constexpr uint32_t M = 2048, N = 2048, K = 1024;
    auto ha = hopper::make_random_vector(size_t(M) * K, 503, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 504, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_h(dA.get(), M, K);
    alignas(64) CUtensorMap tb = make_tmap_h(dB.get(), N, K);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 2048x2048x1024 multicast(A)（Phase 5 对照 103.6T）\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_cluster_2048.csv", {"stages", "min_ms", "tflops"});
    for (uint32_t s : {2u, 3u, 4u, 6u}) {
      auto st = hopper::time_reps(3, 10, [&] {
        dispatch(s, ta, tb, dC.get(), M, N, K);
      });
      const double tf = hopper::tflops(flops, st.min_ms);
      std::printf("  %-7u %12.3f %10.2f\n", s, st.min_ms, tf);
      csv.row({std::to_string(s), std::to_string(st.min_ms), std::to_string(tf)});
    }
  }
  {
    constexpr uint32_t M = 8192, N = 8192, K = 8192;
    auto ha = hopper::make_random_vector(size_t(M) * K, 505, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 506, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_h(dA.get(), M, K);
    alignas(64) CUtensorMap tb = make_tmap_h(dB.get(), N, K);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 8192x8192x8192 multicast(A)（Phase 5 对照 72.2T）\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_cluster_8192.csv", {"stages", "min_ms", "tflops"});
    for (uint32_t s : {2u, 3u, 4u}) {
      auto st = hopper::time_reps(3, 5, [&] {
        dispatch(s, ta, tb, dC.get(), M, N, K);
      });
      const double tf = hopper::tflops(flops, st.min_ms);
      std::printf("  %-7u %12.3f %10.2f\n", s, st.min_ms, tf);
      csv.row({std::to_string(s), std::to_string(st.min_ms), std::to_string(tf)});
    }
  }
  return 0;
}
