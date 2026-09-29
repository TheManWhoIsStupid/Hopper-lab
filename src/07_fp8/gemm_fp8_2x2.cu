// Phase 7d: fp8 × 2×2 cluster 双 multicast —— multicast 线的最后一站。
//
//   拓扑: __cluster_dims__(2,2,1)，cluster 覆盖 bm×bn 的 2×2 tile 块。
//     rank = cbx + 2*cby（cbx = blockIdx.x&1, cby = blockIdx.y&1；6 系列
//     实证 x 是 ctarank 线性化的最快维）。grid 两维都必须是偶数。
//   发射职责（完美匹配，owner ∈ mask，每个 CTA 每拍恰好发一条 multicast）:
//     rank0 → A(cby=0) mask{0,1}=0x3    rank3 → A(cby=1) mask{2,3}=0xC
//     rank2 → B(cbx=0) mask{0,2}=0x5    rank1 → B(cbx=1) mask{1,3}=0xA
//   fetch 模型（8192³）: A 组内 2 共享 → 4GB + B 组内 2 共享 → 2GB = 6GB
//     （7b 单 CTA 12GB 的 **-50%**，7c 单侧 8GB 的 -25%）。
//   屏障协议（6c/6d/7c 教训的全推广）:
//     * full[s] 每拍 expect 24KB = A mcast 16KB + B mcast 8KB（两条不同
//       owner 的 multicast 向同一 barrier 记账，mbarrier tx 是累计语义）；
//     * empty_a/empty_b 只在 owner 端 init 512（mask 内两 CTA 的消费者
//       全体远程 arrive），producer 只等自己覆写的那条；
//     * armed 从"leader 专属"推广为环形握手: 每个 CTA 都是某条 mcast 的
//       非 owner 接收方，先向 partner（覆盖自己的 owner）报 armed，
//       **再**等自己的——arrive 必须在 wait 之前，否则 4-环死锁；
//     * 收尾 cluster_sync 在 if/else 汇合后的单一调用点（.aligned PC
//       一致性 + 迟到远程 arrive 的 release 兜底），epilogue 在其后。
#include <cuda.h>
#include <cuda_fp8.h>
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

constexpr uint32_t kBM = 128, kBN = 64, kBK = 128;  // BK=128: fp8 SW128 原子满宽
constexpr uint32_t kTileA = kBM * kBK;             // 16KB（1B/元素）
constexpr uint32_t kTileB = kBN * kBK;             // 8KB
constexpr uint32_t kStageBytes = kTileA + kTileB;  // 24KB，与 7c 相同
constexpr uint32_t kBlock = 384;                   // 1 producer + 2 consumer WG
constexpr uint32_t kConsumers = 256;

template <uint32_t S>
__global__ void __cluster_dims__(2, 2, 1) __launch_bounds__(kBlock)
gemm_fp8_2x2_kernel(const __grid_constant__ CUtensorMap tmap_a,
                    const __grid_constant__ CUtensorMap tmap_b,
                    float* __restrict__ D, uint32_t M, uint32_t N, uint32_t K) {
  extern __shared__ __align__(1024) char smem_raw[];
  char* sA = smem_raw;                  // S x 16KB
  char* sB = smem_raw + S * kTileA;     // S x 8KB
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_raw + S * kStageBytes);
  uint64_t* empty_a = full + S;    // 只有 A-owner（rank0/3）的实例被使用
  uint64_t* empty_b = empty_a + S;  // 只有 B-owner（rank1/2）的实例被使用
  uint64_t* armed = empty_b + S;   // 人人都是接收方：四个实例全部在用

  const uint32_t tid = threadIdx.x;
  const uint32_t rank = hopper::cluster_ctarank();  // = cbx + 2*cby
  const uint32_t cbx = rank & 1, cby = rank >> 1;
  const uint32_t bm = blockIdx.y, bn = blockIdx.x;
  const uint32_t base_m = blockIdx.y & ~1u, base_n = blockIdx.x & ~1u;
  const uint32_t kblocks = K / kBK;

  const bool a_owner = (rank == 0 || rank == 3);
  const uint32_t partner[4] = {2, 0, 3, 1};  // 覆盖我的那条 mcast 的 owner

  if (tid == 0) {
    for (uint32_t s = 0; s < S; ++s) {
      hopper::mbarrier_init(&full[s], 1);
      hopper::mbarrier_init(&armed[s], 1);
      // 覆写 guard 计数 512 = mask 内两 CTA 的消费者全体（远程 arrive）
      if (a_owner) {
        hopper::mbarrier_init(&empty_a[s], 2 * kConsumers);
      } else {
        hopper::mbarrier_init(&empty_b[s], 2 * kConsumers);
      }
    }
    hopper::fence_mbarrier_init();  // 远程 arrive 前必须（6a 踩坑）
  }
  __syncthreads();
  hopper::cluster_sync();  // barrier init 对整个 cluster 可见；全体收敛执行

  // 累加器作用域必须跨过收尾 cluster_sync（epilogue 在 barrier 之后）。
  const uint32_t wg = (tid - 128) / 128;  // 仅 consumer 路径使用
  float d[32] = {};

  if (tid < 128) {
    // ---- producer warpgroup: 每个 CTA 拥有并发射一条 multicast ----
    if (tid == 0) {
      for (uint32_t j = 0; j < kblocks; ++j) {
        const uint32_t s = j % S;
        if (j >= S) {
          // 只等自己覆写的那条 empty（A 区 guard 在 A-owner，B 区同理）
          hopper::mbarrier_wait_parity(a_owner ? &empty_a[s] : &empty_b[s],
                                       ((j / S) - 1) & 1);
        }
        hopper::mbarrier_arrive_expect_tx(&full[s], kStageBytes);
        // 先向 partner 报到（我的 expect 已落地），再等自己的 armed——
        // 顺序反了就是 4-环死锁（人人等，无人到）
        hopper::mbarrier_arrive_remote(
            hopper::mapa(hopper::smem_u32(&armed[s]), partner[rank]));
        hopper::mbarrier_wait_parity(&armed[s], (j / S) & 1);
        if (a_owner) {
          hopper::tma_load_2d_mcast(sA + s * kTileA, &tmap_a, int32_t(j * kBK),
                                    int32_t((base_m + cby) * kBM), &full[s],
                                    rank == 0 ? 0x3 : 0xC);
        } else {
          hopper::tma_load_2d_mcast(sB + s * kTileB, &tmap_b, int32_t(j * kBK),
                                    int32_t((base_n + cbx) * kBN), &full[s],
                                    rank == 2 ? 0x5 : 0xA);
        }
      }
    }
    // 不 return：落到收尾 cluster_sync（.aligned 要求全 CTA 到场）
  } else {
    // ---- consumer warpgroups: 每 kblock 4 条 k32（与 7c 逐字节相同）----
    const uint32_t a_own = cby == 0 ? 0 : 3;  // 我的 A 区由谁覆写
    const uint32_t b_own = cbx == 0 ? 2 : 1;  // 我的 B 区由谁覆写
    hopper::wgmma_fence();
    for (uint32_t i = 0; i < kblocks; ++i) {
      hopper::mbarrier_wait_parity(&full[i % S], (i / S) & 1);
      const char* sa = sA + (i % S) * kTileA + wg * 8192;
      const char* sb = sB + (i % S) * kTileB;
#pragma unroll
      for (uint32_t kk = 0; kk < kBK / 32; ++kk) {
        const uint64_t da = hopper::gmma_desc_k_sw128(sa + kk * 32, 1024, 0);
        const uint64_t db = hopper::gmma_desc_k_sw128(sb + kk * 32, 1024, 0);
        hopper::wgmma_m64n64k32_f32_e4m3(da, db, d, (i || kk) ? 1u : 0u);
      }
      hopper::wgmma_commit_group();
      if (i >= 1) {
        hopper::wgmma_wait_group<1>();
        const uint32_t s = (i - 1) % S;
        // A/B 两区都由（可能远程的）owner 覆写：两笔 arrive 走 mapa（自报也远程）
        hopper::mbarrier_arrive_remote(
            hopper::mapa(hopper::smem_u32(&empty_a[s]), a_own));
        hopper::mbarrier_arrive_remote(
            hopper::mapa(hopper::smem_u32(&empty_b[s]), b_own));
      }
    }
    hopper::wgmma_wait_group<0>();
  }

  // 收尾同步（6c 的 ULF 教训）：cluster barrier 的 release 语义兜住互连上
  // 迟到的远程 arrive。必须在 if/else 汇合后的单一调用点（.aligned 的 PC
  // 一致性），epilogue 移到 barrier 之后。
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

CUtensorMap make_tmap_fp8(const __nv_fp8_e4m3* dptr, uint32_t rows, uint32_t cols,
                          uint32_t box_rows) {
  alignas(64) CUtensorMap tmap{};
  const cuuint64_t gdim[2] = {cols, rows};
  const cuuint64_t gstride[1] = {cols * sizeof(__nv_fp8_e4m3)};
  const cuuint32_t box[2] = {kBK, box_rows};
  const cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
      const_cast<__nv_fp8_e4m3*>(dptr), gdim, gstride, box, estride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled(fp8) 失败: CUresult %d\n", (int)r);
    std::exit(EXIT_FAILURE);
  }
  return tmap;
}

std::vector<__nv_fp8_e4m3> quantize(const std::vector<float>& f) {
  std::vector<__nv_fp8_e4m3> q(f.size());
  for (size_t i = 0; i < f.size(); ++i) q[i] = __nv_fp8_e4m3(f[i]);
  return q;
}

template <uint32_t S>
void launch_config(const CUtensorMap& ta, const CUtensorMap& tb, float* d,
                   uint32_t M, uint32_t N, uint32_t K) {
  const size_t smem = S * kStageBytes + 4 * S * sizeof(uint64_t);
  CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(gemm_fp8_2x2_kernel<S>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  gemm_fp8_2x2_kernel<S><<<dim3(N / kBN, M / kBM), kBlock, smem>>>(ta, tb, d, M,
                                                                    N, K);
  CUDA_CHECK_LAST();
}

void dispatch(uint32_t stages, const CUtensorMap& ta, const CUtensorMap& tb,
              float* d, uint32_t M, uint32_t N, uint32_t K) {
  // 2x2 cluster 要求 grid 两维均为偶数
  if ((M / kBM) % 2 || (N / kBN) % 2) {
    std::fprintf(stderr, "grid 维度须为偶数 (M/%u=%u, N/%u=%u)\n", kBM, M / kBM,
                 kBN, N / kBN);
    std::exit(1);
  }
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

  // ---- 正确性: 512x512x256（K=256 = 2 个 kblock），S x 3（cluster 时序竞态多 rep）----
  {
    constexpr uint32_t M = 512, N = 512, K = 256;
    auto ha = hopper::make_random_vector(size_t(M) * K, 1001, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 1002, -1.f, 1.f);
    auto a = quantize(ha), b = quantize(hb);
    hopper::DeviceBuffer<__nv_fp8_e4m3> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_fp8(dA.get(), M, K, kBM);
    alignas(64) CUtensorMap tb = make_tmap_fp8(dB.get(), N, K, kBN);

    std::vector<float> ref(size_t(M) * N);  // 从反量化的同一份字节精确重算
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
        auto r = hopper::check_close(dC.download(), ref, 3e-2, 3e-2);
        ok = ok && r.pass;
        if (!r.pass) {
          char tag[48];
          std::snprintf(tag, sizeof(tag), "fp8_2x2 S=%u rep=%u", s, rep);
          hopper::print_report(r, tag);
        }
      }
      std::printf("[CHECK ] fp8_2x2 S=%u x3   : %s\n", s, ok ? "PASS" : "FAIL");
      if (!ok) return 1;
    }
  }

  // ---- 吞吐: 对照 7b fp8 ws / 7c 单侧 mcast ----
  std::filesystem::create_directories("results");
  {
    constexpr uint32_t M = 2048, N = 2048, K = 1024;
    auto a = quantize(hopper::make_random_vector(size_t(M) * K, 1003, -1.f, 1.f));
    auto b = quantize(hopper::make_random_vector(size_t(N) * K, 1004, -1.f, 1.f));
    hopper::DeviceBuffer<__nv_fp8_e4m3> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_fp8(dA.get(), M, K, kBM);
    alignas(64) CUtensorMap tb = make_tmap_fp8(dB.get(), N, K, kBN);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 2048x2048x1024 fp8 2x2 双mcast BM=128（7b ws 209.9T / 7c 单侧 191.6T）\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_fp8_2x2_2048.csv",
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
    auto a = quantize(hopper::make_random_vector(size_t(M) * K, 1005, -1.f, 1.f));
    auto b = quantize(hopper::make_random_vector(size_t(N) * K, 1006, -1.f, 1.f));
    hopper::DeviceBuffer<__nv_fp8_e4m3> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_fp8(dA.get(), M, K, kBM);
    alignas(64) CUtensorMap tb = make_tmap_fp8(dB.get(), N, K, kBN);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 8192x8192x8192 fp8 2x2 双mcast BM=128（7b 满载 174.8T/快窗 260.0T；7c 持平）\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_fp8_2x2_8192.csv",
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
