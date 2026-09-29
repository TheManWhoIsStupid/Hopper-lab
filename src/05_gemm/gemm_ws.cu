// Phase 5b: warp-specialized GEMM —— producer 独立 warpgroup + BM=128。
//
//   384 线程 = 3 个 warpgroup:
//     WG0 (tid 0..127)   producer: 仅 tid0 发射 TMA（无累加器寄存器 =>
//                         脱离 wgmma 的寄存器窗口，ptxas C7520 串行化消失）
//     WG1/WG2 (tid 128..383) consumer: 各跑 m64n64k16，分别负责输出的
//                         行 0-63 / 64-127 —— BM=128，两 WG 共享同一份 B
//
//   tile: BM=128, BN=BK=64; 每 stage: A 16KB + B 8KB = 24KB (SW128)
//   full[s] count=1（producer 登记 24KB）; empty[s] count=256（两 consumer WG
//   全体 arrive）。producer 与 consumer 各跑各的循环，只靠 barrier 交互。
//
//   A 的 TMA box 为 {64,128}：SW128 布局按 8 行原子沿 m 堆叠，行 64 起点 =
//   原子 8 = 偏移 8192（1024B 对齐，硬件可从地址位恢复相位）——consumer wg 的
//   描述符 = sA + wg*8192 + kk*32，SBO 仍 1024。
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "common/bench.h"
#include "common/checker.h"
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
__global__ void __launch_bounds__(kBlock)
gemm_ws_kernel(const __grid_constant__ CUtensorMap tmap_a,
               const __grid_constant__ CUtensorMap tmap_b,
               float* __restrict__ D, uint32_t M, uint32_t N, uint32_t K) {
  extern __shared__ __align__(1024) char smem_raw[];
  char* sA = smem_raw;                  // S x 16KB
  char* sB = smem_raw + S * kTileA;     // S x 8KB
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_raw + S * kStageBytes);
  uint64_t* empty = full + S;

  const uint32_t tid = threadIdx.x;
  const uint32_t bm = blockIdx.y, bn = blockIdx.x;
  const uint32_t kblocks = K / kBK;

  if (tid == 0) {
    for (uint32_t s = 0; s < S; ++s) {
      hopper::mbarrier_init(&full[s], 1);
      hopper::mbarrier_init(&empty[s], kConsumers);
    }
  }
  __syncthreads();

  if (tid < 128) {
    // ---- producer warpgroup: 独立循环，无累加器 ----
    if (tid == 0) {
      for (uint32_t j = 0; j < kblocks; ++j) {
        if (j >= S) {
          hopper::mbarrier_wait_parity(&empty[j % S], ((j / S) - 1) & 1);
        }
        hopper::mbarrier_arrive_expect_tx(&full[j % S], kStageBytes);
        hopper::tma_load_2d(sA + (j % S) * kTileA, &tmap_a, int32_t(j * kBK),
                            int32_t(bm * kBM), &full[j % S]);
        hopper::tma_load_2d(sB + (j % S) * kTileB, &tmap_b, int32_t(j * kBK),
                            int32_t(bn * kBN), &full[j % S]);
      }
    }
    return;  // producer WG 全体退出计算路径
  }

  // ---- consumer warpgroups ----
  const uint32_t wg = (tid - 128) / 128;  // 0: 行 0-63, 1: 行 64-127
  float d[32] = {};
  hopper::wgmma_fence();
  for (uint32_t i = 0; i < kblocks; ++i) {
    hopper::mbarrier_wait_parity(&full[i % S], (i / S) & 1);
    const char* sa = sA + (i % S) * kTileA + wg * 8192;  // 本 WG 的 64 行半区
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
      hopper::mbarrier_arrive(&empty[(i - 1) % S]);
    }
  }
  hopper::wgmma_wait_group<0>();

  const uint32_t warp = (tid - 128) % 128 / 32, lane = tid % 32;
#pragma unroll
  for (uint32_t r = 0; r < 32; ++r) {
    const uint32_t m = 16 * warp + lane / 4 + 8 * ((r / 2) % 2);
    const uint32_t n = 2 * (lane % 4) + r % 2 + 8 * (r / 4);
    D[size_t(bm * kBM + wg * 64 + m) * N + bn * kBN + n] = d[r];
  }
}

CUtensorMap make_tmap_h(const __half* dptr, uint32_t rows, uint32_t cols,
                        uint32_t box_rows) {
  alignas(64) CUtensorMap tmap{};
  const cuuint64_t gdim[2] = {cols, rows};
  const cuuint64_t gstride[1] = {cols * sizeof(__half)};
  const cuuint32_t box[2] = {64, box_rows};
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
  const size_t smem = S * kStageBytes + 2 * S * sizeof(uint64_t);
  CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(gemm_ws_kernel<S>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  gemm_ws_kernel<S><<<dim3(N / kBN, M / kBM), kBlock, smem>>>(ta, tb, d, M, N, K);
  CUDA_CHECK_LAST();
}

void dispatch(uint32_t stages, const CUtensorMap& ta, const CUtensorMap& tb,
              float* d, uint32_t M, uint32_t N, uint32_t K) {
  switch (stages) {
    case 2: launch_config<2>(ta, tb, d, M, N, K); break;
    case 3: launch_config<3>(ta, tb, d, M, N, K); break;
    case 4: launch_config<4>(ta, tb, d, M, N, K); break;
    case 6: launch_config<6>(ta, tb, d, M, N, K); break;
    case 8: launch_config<8>(ta, tb, d, M, N, K); break;
    default: std::fprintf(stderr, "未支持的 stage 数: %u\n", stages); std::exit(1);
  }
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  // ---- 正确性: 512x512x256, S in {2,3,4} ----
  {
    constexpr uint32_t M = 512, N = 512, K = 256;
    auto ha = hopper::make_random_vector(size_t(M) * K, 601, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 602, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_h(dA.get(), M, K, kBM);
    alignas(64) CUtensorMap tb = make_tmap_h(dB.get(), N, K, kBN);

    std::vector<float> ref(size_t(M) * N);
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t n = 0; n < N; ++n) {
        float acc = 0.f;
        for (uint32_t k = 0; k < K; ++k)
          acc += float(a[size_t(m) * K + k]) * float(b[size_t(n) * K + k]);
        ref[size_t(m) * N + n] = acc;
      }

    for (uint32_t s : {2u, 3u, 4u}) {
      dispatch(s, ta, tb, dC.get(), M, N, K);
      auto r = hopper::check_close(dC.download(), ref, 1e-3, 1e-3);
      char tag[32];
      std::snprintf(tag, sizeof(tag), "ws S=%u 512^2x256", s);
      hopper::print_report(r, tag);
      if (!r.pass) return 1;
    }
  }

  // ---- 吞吐: 对照 Phase 5 单 WG（103.6T / 72.2T） ----
  std::filesystem::create_directories("results");
  {
    constexpr uint32_t M = 2048, N = 2048, K = 1024;
    auto ha = hopper::make_random_vector(size_t(M) * K, 603, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 604, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_h(dA.get(), M, K, kBM);
    alignas(64) CUtensorMap tb = make_tmap_h(dB.get(), N, K, kBN);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 2048x2048x1024 warp-specialized BM=128（Phase 5 对照 103.6T）\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_ws_2048.csv", {"stages", "min_ms", "tflops"});
    for (uint32_t s : {2u, 3u, 4u, 6u, 8u}) {
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
    auto ha = hopper::make_random_vector(size_t(M) * K, 605, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 606, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_h(dA.get(), M, K, kBM);
    alignas(64) CUtensorMap tb = make_tmap_h(dB.get(), N, K, kBN);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 8192x8192x8192 warp-specialized BM=128（Phase 5 对照 72.2T）\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_ws_8192.csv", {"stages", "min_ms", "tflops"});
    for (uint32_t s : {2u, 3u, 4u, 6u}) {
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
