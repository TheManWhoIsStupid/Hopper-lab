// Phase 6d: warp-specialized multicast GEMM —— 5b 与 6c 的组合。
//
//   5b 结构: 384 线程 = producer WG（独立 issue 循环，脱离累加器寄存器窗口）
//            + 2 consumer WG（BM=128，各管 64 行半区）
//   6c 拓扑: __cluster_dims__(2,1,1)，同 bm 不同 bn 的相邻 CTA 成对，
//            A 走 multicast（leader 单点发射），B 各自 unicast
//
// 屏障拓扑（6c 语义 + warp-spec 的消费者数）:
//   full[s]    每 CTA, count=1: 各自 tid0 登记 24KB expect
//   empty_a[s] 仅 leader, count=512: 两 CTA 全部 512 个消费者线程 arrive
//              （rank1 远程）—— multicast 覆写两边 A 区，释放须双 CTA 确认
//   empty_b[s] 每 CTA, count=256: 自己的 B 区自己管
//   armed[s]   仅 leader, count=1: rank1 登记 expect 后远程报 armed，
//              rank0 见到才发射（相位敏感的记账竞态，6b/6c 实锤）
//
//   fetch 模型（8192³）: A 经 cluster 共享 16GB + B 8GB = 24GB → 16GB（-33%）
//   对照 5b ws 单机 24GB / 6c 单 WG mcast 拓扑 24GB（BM=64 时 A+B 对半）。
//
// ⚠ 与 5b 的关键差异: producer WG **不提前 return**——barrier.cluster 的
//   .aligned 要求全 CTA 收敛执行（6a 踩坑），收尾 cluster_sync 必须人人到场。
//   producer 循环跑完后直接落到收尾同步上等消费者。
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

constexpr uint32_t kBM = 128, kBN = 64, kBK = 64;
constexpr uint32_t kTileA = kBM * kBK * 2;       // 16KB
constexpr uint32_t kTileB = kBN * kBK * 2;       // 8KB
constexpr uint32_t kStageBytes = kTileA + kTileB;
constexpr uint32_t kBlock = 384;                 // 1 producer + 2 consumer WG
constexpr uint32_t kConsumers = 256;

template <uint32_t S>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(kBlock)
gemm_ws_cluster_kernel(const __grid_constant__ CUtensorMap tmap_a,
                       const __grid_constant__ CUtensorMap tmap_b,
                       float* __restrict__ D, uint32_t M, uint32_t N, uint32_t K) {
  extern __shared__ __align__(1024) char smem_raw[];
  char* sA = smem_raw;                  // S x 16KB
  char* sB = smem_raw + S * kTileA;     // S x 8KB
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_raw + S * kStageBytes);
  uint64_t* empty_a = full + S;   // 仅 leader 的实例被使用（偏移两 CTA 一致）
  uint64_t* empty_b = empty_a + S;
  uint64_t* armed = empty_b + S;

  const uint32_t tid = threadIdx.x;
  const uint32_t rank = hopper::cluster_ctarank();  // = blockIdx.x % 2
  const uint32_t bm = blockIdx.y, bn = blockIdx.x;
  const uint32_t kblocks = K / kBK;

  if (tid == 0) {
    for (uint32_t s = 0; s < S; ++s) {
      hopper::mbarrier_init(&full[s], 1);
      hopper::mbarrier_init(&empty_b[s], kConsumers);
      if (rank == 0) {
        hopper::mbarrier_init(&empty_a[s], 2 * kConsumers);
        hopper::mbarrier_init(&armed[s], 1);
      }
    }
    hopper::fence_mbarrier_init();  // 远程 arrive 前必须（6a 踩坑）
  }
  __syncthreads();
  hopper::cluster_sync();  // barrier init 对整个 cluster 可见；全体收敛执行

  // 累加器作用域必须跨过收尾 cluster_sync（epilogue 在 barrier 之后）。
  // 寄存器分配全核统一，producer 线程背着这 32 个寄存器不亏（5b 同款）。
  const uint32_t wg = (tid - 128) / 128;  // 仅 consumer 路径使用
  float d[32] = {};

  if (tid < 128) {
    // ---- producer warpgroup: 独立循环，leader/follower 逻辑都在这里 ----
    if (tid == 0) {
      for (uint32_t j = 0; j < kblocks; ++j) {
        const uint32_t s = j % S;
        if (j >= S) {
          hopper::mbarrier_wait_parity(&empty_b[s], ((j / S) - 1) & 1);
          if (rank == 0) {
            hopper::mbarrier_wait_parity(&empty_a[s], ((j / S) - 1) & 1);
          }
        }
        hopper::mbarrier_arrive_expect_tx(&full[s], kStageBytes);
        if (rank == 0) {
          // rank1 的 expect 登记落地后才能发射（相位敏感记账）
          hopper::mbarrier_wait_parity(&armed[s], (j / S) & 1);
          hopper::tma_load_2d_mcast(sA + s * kTileA, &tmap_a, int32_t(j * kBK),
                                    int32_t(bm * kBM), &full[s], 0x3);
          hopper::tma_load_2d(sB + s * kTileB, &tmap_b, int32_t(j * kBK),
                              int32_t(bn * kBN), &full[s]);
        } else {
          hopper::mbarrier_arrive_remote(hopper::mapa(hopper::smem_u32(&armed[s]), 0));
          hopper::tma_load_2d(sB + s * kTileB, &tmap_b, int32_t(j * kBK),
                              int32_t(bn * kBN), &full[s]);
        }
      }
    }
    // 不 return：落到收尾 cluster_sync（.aligned 要求全 CTA 到场）
  } else {
    // ---- consumer warpgroups ----
    hopper::wgmma_fence();
    for (uint32_t i = 0; i < kblocks; ++i) {
      hopper::mbarrier_wait_parity(&full[i % S], (i / S) & 1);
      const char* sa = sA + (i % S) * kTileA + wg * 8192;
      const char* sb = sB + (i % S) * kTileB;
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
        // A 区释放须让 leader 知道（multicast 覆写两边）；B 区各自本地
        hopper::mbarrier_arrive_remote(hopper::mapa(hopper::smem_u32(&empty_a[s]), 0));
        hopper::mbarrier_arrive(&empty_b[s]);
      }
    }
    hopper::wgmma_wait_group<0>();
  }

  // 收尾同步（6c 的 ULF 教训）: 最后的远程 arrive（rank1 -> rank0 的
  // empty_a/armed）可能还在互连上；先退出的 CTA smem 被回收，迟到 arrive
  // 打到已释放地址 = ULF。cluster barrier 的 release 语义兜住这一切。
  //
  // ⚠ .aligned 要求全 CTA 执行**同一条** barrier 指令——producer/consumer
  // 两个分支各自调用 = 两个不同 PC，违规（6a 挂死的变体）。必须在 if/else
  // 汇合后的单一调用点执行（warp 不跨 tid<128 边界，分支内每 warp 全 32
  // lane 同路，汇合点全员收敛）。epilogue 移到 barrier 之后（D 无跨 CTA
  // 重叠，晚写无害，还顺带拉大 CTA 退出间隔）。
  hopper::cluster_sync();

  if (tid >= 128) {
    const uint32_t warp = (tid - 128) % 128 / 32, lane = tid % 32;
#pragma unroll
    for (uint32_t r = 0; r < 32; ++r) {
      const uint32_t m = 16 * warp + lane / 4 + 8 * ((r / 2) % 2);
      const uint32_t n = 2 * (lane % 4) + r % 2 + 8 * (r / 4);
      D[size_t(bm * kBM + wg * 64 + m) * N + bn * kBN + n] = d[r];
    }
  }
}

CUtensorMap make_tmap_a(const __half* dptr, uint32_t rows, uint32_t cols) {
  alignas(64) CUtensorMap tmap{};
  const cuuint64_t gdim[2] = {cols, rows};
  const cuuint64_t gstride[1] = {cols * sizeof(__half)};
  const cuuint32_t box[2] = {64, 128};  // A 的 box 高 128（BM=128）
  const cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, const_cast<__half*>(dptr), gdim,
      gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled(A) 失败: CUresult %d\n", (int)r);
    std::exit(EXIT_FAILURE);
  }
  return tmap;
}

CUtensorMap make_tmap_b(const __half* dptr, uint32_t rows, uint32_t cols) {
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
    std::fprintf(stderr, "cuTensorMapEncodeTiled(B) 失败: CUresult %d\n", (int)r);
    std::exit(EXIT_FAILURE);
  }
  return tmap;
}

template <uint32_t S>
void launch_config(const CUtensorMap& ta, const CUtensorMap& tb, float* d,
                   uint32_t M, uint32_t N, uint32_t K) {
  const size_t smem = S * kStageBytes + 4 * S * sizeof(uint64_t);
  CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(gemm_ws_cluster_kernel<S>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  gemm_ws_cluster_kernel<S><<<dim3(N / kBN, M / kBM), kBlock, smem>>>(ta, tb, d, M,
                                                                       N, K);
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

  // ---- 正确性: 512x512x256, S in {2,3,4} x3（cluster 时序竞态要多 rep） ----
  {
    constexpr uint32_t M = 512, N = 512, K = 256;
    auto ha = hopper::make_random_vector(size_t(M) * K, 701, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 702, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_a(dA.get(), M, K);
    alignas(64) CUtensorMap tb = make_tmap_b(dB.get(), N, K);

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
      for (uint32_t rep = 0; rep < 3; ++rep) {
        dispatch(s, ta, tb, dC.get(), M, N, K);
        auto r = hopper::check_close(dC.download(), ref, 1e-3, 1e-3);
        ok = ok && r.pass;
        if (!r.pass) {
          char tag[48];
          std::snprintf(tag, sizeof(tag), "ws_cluster S=%u rep=%u", s, rep);
          hopper::print_report(r, tag);
        }
      }
      std::printf("[CHECK ] ws_cluster S=%u x3      : %s\n", s,
                  ok ? "PASS" : "FAIL");
      if (!ok) return 1;
    }
  }

  // ---- 吞吐: 对照 5b ws（117.6T / 73.1T）与 6c（95.9T / 72.5T） ----
  std::filesystem::create_directories("results");
  {
    constexpr uint32_t M = 2048, N = 2048, K = 1024;
    auto ha = hopper::make_random_vector(size_t(M) * K, 703, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 704, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_a(dA.get(), M, K);
    alignas(64) CUtensorMap tb = make_tmap_b(dB.get(), N, K);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 2048x2048x1024 ws+mcast BM=128（5b 对照 117.6T / 6c 对照 95.9T）\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_ws_cluster_2048.csv",
                          {"stages", "min_ms", "tflops"});
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
    auto ha = hopper::make_random_vector(size_t(M) * K, 705, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 706, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_a(dA.get(), M, K);
    alignas(64) CUtensorMap tb = make_tmap_b(dB.get(), N, K);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 8192x8192x8192 ws+mcast BM=128（5b 对照 73.1T / 6c 对照 72.5T）\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_ws_cluster_8192.csv",
                          {"stages", "min_ms", "tflops"});
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
