// Phase 8a: fp8 persistent GEMM —— 固定驻留 grid + group-M tile 调度。
//
//   结构: 7b 的 ws 流水线（1 producer WG + 2 consumer WG，fp8 通路）原样，
//   但 grid 是 1D 固定驻留集（occupancy × SM 数），CTA 用 for-loop 领取
//   tile。两个 persistent 专属改动：
//   * **kblock→stage 环跨 tile 连续**（j 是全 kernel 全局计数器）——
//     tile t 的 epilogue 与 tile t+1 的 TMA 预取由 full/empty 屏障自然
//     重叠，这是 persistent 的核心红利（epilogue 不再是每波末尾的串行尾巴）。
//   * **tile 末尾补释放最后一级 stage**（empty arrive at (j-1)%S）——
//     7b 单 tile 版 kernel 跑完即退不用管；persistent 里环要绕回，漏放
//     这一级 = 下一圈 producer 死等（S 级之后精确挂死）。
//   * epilogue 暂存改用**专用区**（2 WG × 16KB，不复用 sA）——producer
//     正在往 sA 预取下一 tile，复用即竞态；顺带 5c 的跨 WG named_barrier(3)
//     不再需要（各 WG 只等自己的 wgmma 排空）。
//
//   tile 调度（group-M swizzle，CUTLASS TileScheduler 同款）:
//     每组 G 个 bm 行，组内 bn 外层。并发窗口（156 CTA）覆盖 G × (156/G)
//     紧凑矩形——A/B tile 双双落进 L2。G=1 退化为 CUDA 自然行波序。
//     8192³ DRAM 读模型: B 行波重取 64MB×(64/G)，G=8 → -87%（4.28GB → ~0.6GB）。
//
//   对照（7b 同日满载）: 2048³ 209.9T / 8192³ 174.3T，DRAM 读 4.28GB。
#include <cuda.h>
#include <cuda_fp8.h>
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

constexpr uint32_t kBM = 128, kBN = 64, kBK = 128;  // BK=128: fp8 SW128 原子满宽
constexpr uint32_t kTileA = kBM * kBK;             // 16KB（1B/元素）
constexpr uint32_t kTileB = kBN * kBK;             // 8KB
constexpr uint32_t kStageBytes = kTileA + kTileB;  // 24KB
constexpr uint32_t kEpBytes = 2 * 16384;           // epilogue 暂存 2 WG × 16KB
constexpr uint32_t kBlock = 384;                   // 1 producer + 2 consumer WG
constexpr uint32_t kConsumers = 256;

__device__ __forceinline__ void named_barrier(uint32_t id, uint32_t count) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}

// tile 序号 -> (bm, bn)。group-M: 每组 G 行，组内 bn 外层（t 先跨行后沿列）。
// G=1 时 = 行优先 x-fastest（CUDA 自然波序），与非 persistent 基线同构。
__device__ __forceinline__ void tile_to_bmn(uint32_t tile, uint32_t num_bm,
                                            uint32_t num_bn, uint32_t G,
                                            uint32_t& bm, uint32_t& bn) {
  const uint32_t tiles_per_group = G * num_bn;
  const uint32_t g = tile / tiles_per_group;
  const uint32_t first_bm = g * G;
  const uint32_t gsize = min(num_bm - first_bm, G);
  const uint32_t t = tile - g * tiles_per_group;
  bm = first_bm + t % gsize;
  bn = t / gsize;
}

template <uint32_t S, bool kTmaStore>
__global__ void __launch_bounds__(kBlock)
gemm_fp8_persist_kernel(const __grid_constant__ CUtensorMap tmap_a,
                        const __grid_constant__ CUtensorMap tmap_b,
                        const __grid_constant__ CUtensorMap tmap_d,
                        float* __restrict__ D, uint32_t M, uint32_t N,
                        uint32_t K, uint32_t group_m) {
  extern __shared__ __align__(1024) char smem_raw[];
  char* sA = smem_raw;                    // S x 16KB
  char* sB = smem_raw + S * kTileA;       // S x 8KB
  char* sEp = smem_raw + S * kStageBytes;  // 2 WG x 16KB（epilogue 专用暂存）
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_raw + S * kStageBytes + kEpBytes);
  uint64_t* empty = full + S;

  const uint32_t tid = threadIdx.x;
  const uint32_t num_bm = M / kBM, num_bn = N / kBN;
  const uint32_t total_tiles = num_bm * num_bn;
  const uint32_t kblocks = K / kBK;

  if (tid == 0) {
    for (uint32_t s = 0; s < S; ++s) {
      hopper::mbarrier_init(&full[s], 1);
      hopper::mbarrier_init(&empty[s], kConsumers);
    }
  }
  __syncthreads();

  const uint32_t wg = (tid - 128) / 128;  // 仅 consumer 路径使用
  if (tid < 128) {
    // ---- producer warpgroup: 全局连续 stage 环，跨 tile 不回头 ----
    if (tid == 0) {
      uint32_t j = 0;  // 全 kernel kblock 计数器（stage = j % S）
      for (uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        uint32_t bm, bn;
        tile_to_bmn(tile, num_bm, num_bn, group_m, bm, bn);
        for (uint32_t kb = 0; kb < kblocks; ++kb, ++j) {
          const uint32_t s = j % S;
          if (j >= S) {
            hopper::mbarrier_wait_parity(&empty[s], ((j / S) - 1) & 1);
          }
          hopper::mbarrier_arrive_expect_tx(&full[s], kStageBytes);
          hopper::tma_load_2d(sA + s * kTileA, &tmap_a, int32_t(kb * kBK),
                              int32_t(bm * kBM), &full[s]);
          hopper::tma_load_2d(sB + s * kTileB, &tmap_b, int32_t(kb * kBK),
                              int32_t(bn * kBN), &full[s]);
        }
      }
    }
  } else {
    // ---- consumer warpgroups: 每 tile 一组累加器，环连续 ----
    uint32_t j = 0;
    const uint32_t warp = (tid - 128) % 128 / 32, lane = tid % 32;
    for (uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
      uint32_t bm, bn;
      tile_to_bmn(tile, num_bm, num_bn, group_m, bm, bn);
      float d[32] = {};
      hopper::wgmma_fence();  // d 的寄存器写在每 tile 开头，逐 tile 重新 fence
      for (uint32_t kb = 0; kb < kblocks; ++kb, ++j) {
        hopper::mbarrier_wait_parity(&full[j % S], (j / S) & 1);
        const char* sa = sA + (j % S) * kTileA + wg * 8192;  // 本 WG 的 64 行半区
        const char* sb = sB + (j % S) * kTileB;
#pragma unroll
        for (uint32_t kk = 0; kk < kBK / 32; ++kk) {
          const uint64_t da = hopper::gmma_desc_k_sw128(sa + kk * 32, 1024, 0);
          const uint64_t db = hopper::gmma_desc_k_sw128(sb + kk * 32, 1024, 0);
          hopper::wgmma_m64n64k32_f32_e4m3(da, db, d, (kb || kk) ? 1u : 0u);
        }
        hopper::wgmma_commit_group();
        if (kb >= 1) {
          hopper::wgmma_wait_group<1>();
          hopper::mbarrier_arrive(&empty[(j - 1) % S]);
        }
      }
      hopper::wgmma_wait_group<0>();
      // persistent 专属：尾级 stage 补释放（否则下一圈 producer 死等）
      hopper::mbarrier_arrive(&empty[(j - 1) % S]);

      if (kTmaStore) {
        // TMA store epilogue：暂存到专用区（布局与 5c/7b 逐字节相同）
        char* stage = sEp + wg * 16384;
#pragma unroll
        for (uint32_t r = 0; r < 32; r += 2) {
          const uint32_t m = 16 * warp + lane / 4 + 8 * ((r / 2) % 2);
          const uint32_t n = 2 * (lane % 4) + r % 2 + 8 * (r / 4);
          const uint32_t nl = n % 32;
          char* p = stage + (n / 32) * 8192 + m * 128 +
                    ((nl / 4) ^ (m % 8)) * 16 + (nl % 4) * 4;
          *reinterpret_cast<float2*>(p) = make_float2(d[r], d[r + 1]);
        }
        named_barrier(1 + wg, 128);  // 本 WG 暂存写齐（专用区无需跨 WG 同步）
        hopper::fence_proxy_async_shared_cta();
        if ((tid - 128) % 128 == 0) {
          const int32_t row = int32_t(bm * kBM + wg * 64);
          hopper::tma_store_2d(stage, &tmap_d, int32_t(bn * kBN), row);
          hopper::tma_store_2d(stage + 8192, &tmap_d, int32_t(bn * kBN + 32),
                               row);
          hopper::tma_store_commit_group();
          hopper::tma_store_wait_read<0>();  // 下一 tile 覆写暂存前必须排空读
        }
      } else {
#pragma unroll
        for (uint32_t r = 0; r < 32; ++r) {
          const uint32_t m = 16 * warp + lane / 4 + 8 * ((r / 2) % 2);
          const uint32_t n = 2 * (lane % 4) + r % 2 + 8 * (r / 4);
          D[size_t(bm * kBM + wg * 64 + m) * N + bn * kBN + n] = d[r];
        }
      }
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

CUtensorMap make_tmap_f(const float* dptr, uint32_t rows, uint32_t cols) {
  alignas(64) CUtensorMap tmap{};
  const cuuint64_t gdim[2] = {cols, rows};
  const cuuint64_t gstride[1] = {cols * sizeof(float)};
  const cuuint32_t box[2] = {32, 64};
  const cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<float*>(dptr), gdim,
      gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled(D) 失败: CUresult %d\n", (int)r);
    std::exit(EXIT_FAILURE);
  }
  return tmap;
}

std::vector<__nv_fp8_e4m3> quantize(const std::vector<float>& f) {
  std::vector<__nv_fp8_e4m3> q(f.size());
  for (size_t i = 0; i < f.size(); ++i) q[i] = __nv_fp8_e4m3(f[i]);
  return q;
}

// 驻留 grid = occupancy × SM 数（上限 total_tiles）
template <uint32_t S, bool kTmaStore>
void launch_config(const CUtensorMap& ta, const CUtensorMap& tb,
                   const CUtensorMap& td, float* d, uint32_t M, uint32_t N,
                   uint32_t K, uint32_t group_m) {
  const size_t smem = S * kStageBytes + kEpBytes + 2 * S * sizeof(uint64_t);
  CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(gemm_fp8_persist_kernel<S, kTmaStore>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int sms = 0, occ = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, gemm_fp8_persist_kernel<S, kTmaStore>, kBlock, smem));
  const uint32_t total = (M / kBM) * (N / kBN);
  const uint32_t grid = std::min(total, uint32_t(sms * occ));
  gemm_fp8_persist_kernel<S, kTmaStore><<<grid, kBlock, smem>>>(
      ta, tb, td, d, M, N, K, group_m);
  CUDA_CHECK_LAST();
}

void dispatch(uint32_t stages, bool tma, const CUtensorMap& ta,
              const CUtensorMap& tb, const CUtensorMap& td, float* d,
              uint32_t M, uint32_t N, uint32_t K, uint32_t group_m) {
  if (group_m == 0) {
    std::fprintf(stderr, "group_m 须 >= 1\n");
    std::exit(1);
  }
  switch (stages) {
    case 2:
      tma ? launch_config<2, true>(ta, tb, td, d, M, N, K, group_m)
          : launch_config<2, false>(ta, tb, td, d, M, N, K, group_m);
      break;
    case 4:
      tma ? launch_config<4, true>(ta, tb, td, d, M, N, K, group_m)
          : launch_config<4, false>(ta, tb, td, d, M, N, K, group_m);
      break;
    case 6:
      tma ? launch_config<6, true>(ta, tb, td, d, M, N, K, group_m)
          : launch_config<6, false>(ta, tb, td, d, M, N, K, group_m);
      break;
    case 8:
      tma ? launch_config<8, true>(ta, tb, td, d, M, N, K, group_m)
          : launch_config<8, false>(ta, tb, td, d, M, N, K, group_m);
      break;
    default: std::fprintf(stderr, "未支持的 stage 数: %u\n", stages); std::exit(1);
  }
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  // ---- 正确性: 512x512x256（单波） + 2048x2048x256（多波，检验跨 tile 环）----
  {
    struct Shape { uint32_t m, n, k; const char* tag; };
    for (Shape sh : {Shape{512, 512, 256, "512"}, Shape{2048, 2048, 256, "2048 多波"}}) {
      const uint32_t M = sh.m, N = sh.n, K = sh.k;
      auto ha = hopper::make_random_vector(size_t(M) * K, 1001, -1.f, 1.f);
      auto hb = hopper::make_random_vector(size_t(N) * K, 1002, -1.f, 1.f);
      auto a = quantize(ha), b = quantize(hb);
      hopper::DeviceBuffer<__nv_fp8_e4m3> dA(a.size()), dB(b.size());
      hopper::DeviceBuffer<float> dC(size_t(M) * N);
      dA.upload(a);
      dB.upload(b);
      alignas(64) CUtensorMap ta = make_tmap_fp8(dA.get(), M, K, kBM);
      alignas(64) CUtensorMap tb = make_tmap_fp8(dB.get(), N, K, kBN);
      alignas(64) CUtensorMap td = make_tmap_f(dC.get(), M, N);

      std::vector<float> ref(size_t(M) * N);
      for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
          float acc = 0.f;
          for (uint32_t k = 0; k < K; ++k)
            acc += float(a[size_t(m) * K + k]) * float(b[size_t(n) * K + k]);
          ref[size_t(m) * N + n] = acc;
        }

      for (uint32_t s : {2u, 4u, 8u}) {
        for (uint32_t g : {1u, 8u}) {
          bool ok = true;
          for (uint32_t rep = 0; rep < 3; ++rep) {
            dispatch(s, true, ta, tb, td, dC.get(), M, N, K, g);
            auto r = hopper::check_close(dC.download(), ref, 3e-2, 3e-2);
            ok = ok && r.pass;
            if (!r.pass) {
              char tag[64];
              std::snprintf(tag, sizeof(tag), "persist %s S=%u G=%u rep=%u",
                            sh.tag, s, g, rep);
              hopper::print_report(r, tag);
            }
          }
          std::printf("[CHECK ] persist %s S=%u G=%-2u x3: %s\n", sh.tag, s, g,
                      ok ? "PASS" : "FAIL");
          if (!ok) return 1;
        }
      }
    }
  }

  // ---- 吞吐: S × G 扫描（对照 7b 同日满载）----
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
    alignas(64) CUtensorMap td = make_tmap_f(dC.get(), M, N);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 2048x2048x1024 fp8 persistent（7b 同日满载 209.9T）\n");
    std::printf("  %-7s %-7s %12s %10s\n", "stages", "group_m", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_fp8_persist_2048.csv",
                          {"stages", "group_m", "min_ms", "tflops"});
    for (uint32_t s : {2u, 4u, 6u, 8u}) {
      for (uint32_t g : {1u, 4u, 8u, 16u}) {
        auto st = hopper::time_reps(3, 10, [&] {
          dispatch(s, true, ta, tb, td, dC.get(), M, N, K, g);
        });
        const double tf = hopper::tflops(flops, st.min_ms);
        std::printf("  %-7u %-7u %12.3f %10.2f\n", s, g, st.min_ms, tf);
        csv.row({std::to_string(s), std::to_string(g), std::to_string(st.min_ms),
                 std::to_string(tf)});
      }
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
    alignas(64) CUtensorMap td = make_tmap_f(dC.get(), M, N);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 8192x8192x8192 fp8 persistent（7b 同日满载 174.3T / 快窗 260.0T）\n");
    std::printf("  %-7s %-7s %12s %10s\n", "stages", "group_m", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_fp8_persist_8192.csv",
                          {"stages", "group_m", "min_ms", "tflops"});
    for (uint32_t s : {4u, 6u, 8u}) {
      for (uint32_t g : {1u, 8u, 16u}) {
        auto st = hopper::time_reps(3, 5, [&] {
          dispatch(s, true, ta, tb, td, dC.get(), M, N, K, g);
        });
        const double tf = hopper::tflops(flops, st.min_ms);
        std::printf("  %-7u %-7u %12.3f %10.2f\n", s, g, st.min_ms, tf);
        csv.row({std::to_string(s), std::to_string(g), std::to_string(st.min_ms),
                 std::to_string(tf)});
      }
    }
  }
  return 0;
}
