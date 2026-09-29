// Phase 7d 消融负控: (2,2) cluster × 全 unicast —— 拓扑税的隔离测量。
//
//   与 gemm_fp8_2x2.cu 唯一的实验变量: 保持 __cluster_dims__(2,2,1) 和
//   首尾 cluster_sync，但主循环零跨 CTA 耦合——A/B 全部本地 unicast
//   （tma_load_2d × 2），empty_a/empty_b 全本地（count 256），无 armed。
//   用于把 "4-CTA 共驻调度税" 与 "multicast 传输/协议开销" 解耦：
//   实测它与双 mcast 版只差 < 2T（114.6 vs 112.7T，8192³ 满载），
//   即 (2,2) 的 -34% 惩罚全部来自拓扑本身。
//
//   正确性/吞吐协议与 7d 正主一致，CSV 落 gemm_fp8_2x2_uni_*.csv。
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
constexpr uint32_t kStageBytes = kTileA + kTileB;  // 24KB
constexpr uint32_t kBlock = 384;                   // 1 producer + 2 consumer WG
constexpr uint32_t kConsumers = 256;

template <uint32_t S>
__global__ void __cluster_dims__(2, 2, 1) __launch_bounds__(kBlock)
gemm_fp8_2x2_uni_kernel(const __grid_constant__ CUtensorMap tmap_a,
                        const __grid_constant__ CUtensorMap tmap_b,
                        float* __restrict__ D, uint32_t M, uint32_t N, uint32_t K) {
  extern __shared__ __align__(1024) char smem_raw[];
  char* sA = smem_raw;                  // S x 16KB
  char* sB = smem_raw + S * kTileA;     // S x 8KB
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_raw + S * kStageBytes);
  uint64_t* empty_a = full + S;     // 本地 guard（A 区自取自用）
  uint64_t* empty_b = empty_a + S;  // 本地 guard（B 区自取自用）

  const uint32_t tid = threadIdx.x;
  const uint32_t bm = blockIdx.y, bn = blockIdx.x;
  const uint32_t kblocks = K / kBK;

  if (tid == 0) {
    for (uint32_t s = 0; s < S; ++s) {
      hopper::mbarrier_init(&full[s], 1);
      hopper::mbarrier_init(&empty_a[s], kConsumers);
      hopper::mbarrier_init(&empty_b[s], kConsumers);
    }
    // 无远程 arrive，cluster 可见性 fence 可省；init 后仍需 CTA 内同步
  }
  __syncthreads();
  hopper::cluster_sync();  // 共驻同步本身是测量对象的一部分，保留

  // 累加器作用域必须跨过收尾 cluster_sync（epilogue 在 barrier 之后）。
  const uint32_t wg = (tid - 128) / 128;  // 仅 consumer 路径使用
  float d[32] = {};

  if (tid < 128) {
    // ---- producer warpgroup: 纯本地双 unicast（7b 同款，无任何跨 CTA 协议）----
    if (tid == 0) {
      for (uint32_t j = 0; j < kblocks; ++j) {
        const uint32_t s = j % S;
        if (j >= S) {
          hopper::mbarrier_wait_parity(&empty_a[s], ((j / S) - 1) & 1);
          hopper::mbarrier_wait_parity(&empty_b[s], ((j / S) - 1) & 1);
        }
        hopper::mbarrier_arrive_expect_tx(&full[s], kStageBytes);
        hopper::tma_load_2d(sA + s * kTileA, &tmap_a, int32_t(j * kBK),
                            int32_t(bm * kBM), &full[s]);
        hopper::tma_load_2d(sB + s * kTileB, &tmap_b, int32_t(j * kBK),
                            int32_t(bn * kBN), &full[s]);
      }
    }
    // 不 return：落到收尾 cluster_sync（.aligned 要求全 CTA 到场）
  } else {
    // ---- consumer warpgroups: 与 7d 正主逐字节相同 ----
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
        hopper::mbarrier_arrive(&empty_a[s]);
        hopper::mbarrier_arrive(&empty_b[s]);
      }
    }
    hopper::wgmma_wait_group<0>();
  }

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
  const size_t smem = S * kStageBytes + 3 * S * sizeof(uint64_t);
  CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(gemm_fp8_2x2_uni_kernel<S>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  gemm_fp8_2x2_uni_kernel<S><<<dim3(N / kBN, M / kBM), kBlock, smem>>>(ta, tb, d, M,
                                                                        N, K);
  CUDA_CHECK_LAST();
}

void dispatch(uint32_t stages, const CUtensorMap& ta, const CUtensorMap& tb,
              float* d, uint32_t M, uint32_t N, uint32_t K) {
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

  // ---- 正确性: 512x512x256（K=256 = 2 个 kblock），S x 3 ----
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
        auto r = hopper::check_close(dC.download(), ref, 3e-2, 3e-2);
        ok = ok && r.pass;
        if (!r.pass) {
          char tag[48];
          std::snprintf(tag, sizeof(tag), "fp8_2x2_uni S=%u rep=%u", s, rep);
          hopper::print_report(r, tag);
        }
      }
      std::printf("[CHECK ] fp8_2x2_uni S=%u x3: %s\n", s, ok ? "PASS" : "FAIL");
      if (!ok) return 1;
    }
  }

  // ---- 吞吐: 拓扑税对照（vs 7b 无 cluster / 7d 双 mcast）----
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
    std::printf("\n[GEMM  ] 2048x2048x1024 fp8 2x2 全unicast（拓扑税负控；7b 209.9T / 7d 双播 153.4T）\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_fp8_2x2_uni_2048.csv",
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
    std::printf("\n[GEMM  ] 8192x8192x8192 fp8 2x2 全unicast（7b 174.8T / 7d 双播 112.7T）\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_fp8_2x2_uni_8192.csv",
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
