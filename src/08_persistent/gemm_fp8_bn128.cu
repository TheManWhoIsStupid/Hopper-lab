// Phase 8b: BN=128 大 tile（m64n128k32）× persistent 底盘 + 动态 tile 队列。
//
//   8a 结论是"流量不是时间的货币"，唯一没动过的轴是计算密度。BN=128 后：
//   * 每 kblock 指令数不变（2 WG × 4 条 k32），FLOP 翻倍（m64n128k32 单条
//     524K vs 262K）——issue/barrier 摊到的 FLOP 全部翻倍
//   * 算力强度 87 → 131 FLOP/B（L2 口径），L2 compulsory 12.6 → 8GB（8192³）
//   * stage 24 → 32KB（B tile 128 宽），sEp 2×32KB，S≤5（224KB 贴 227 上限）
//
//   动态 tile 队列（sched=1）修 8a 的静态量化损失——BN=128 会放大它：
//   2048 tile 数 512→256，256/78 = 3.28 tiles/CTA，静态轮转 makespan +22%
//   理论。代价是领序发散 ⇒ G=8 紧凑窗口退化（L2 局部性换均衡），ncu 可测。
//
//   队列实现（关键：timed path 零 memset）：
//   * g_tile_ctr/g_done_ctr 是模块级 __device__ 变量，装载时零初始化；
//   * **自复位**：所有 producer 退出 tile 循环后 atomicAdd g_done_ctr，
//     最后一个 CTA 把两个计数器都清零——下一次 launch 从 0 开始，
//     任何 host 侧 reset 都不进计时区
//   * tile 序号经 s_go[2] mbarrier 环从 producer tid0 交接给全体消费者
//     （写 s_tile 后 arrive，消费者 wait 后读）。环安全性：producer 的
//     领先受 empty[] 门控 ≤ fetch 领先 + 1 tile < 2 tiles，距 2 的同 buffer
//     重叠不可能发生。超领（tile >= total）也走 s_go 交接——消费者看到
//     越界值即退出，不死等一个永远不会来的 full
//
//   对照（同日满载）: 7b 174.59T / 8a persist 174.5T，DRAM G=8 605MB。
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

constexpr uint32_t kBM = 128, kBN = 128, kBK = 128;
constexpr uint32_t kTileA = kBM * kBK;             // 16KB
constexpr uint32_t kTileB = kBN * kBK;             // 16KB（BN=128 翻倍）
constexpr uint32_t kStageBytes = kTileA + kTileB;  // 32KB
constexpr uint32_t kEpBytes = 2 * 32768;           // epilogue 暂存 2 WG × 32KB
constexpr uint32_t kBlock = 384;
constexpr uint32_t kConsumers = 256;

// 动态 tile 队列（模块级，装载零初始化，kernel 尾自复位——见文件头）
__device__ uint32_t g_tile_ctr = 0;
__device__ uint32_t g_done_ctr = 0;

__device__ __forceinline__ void named_barrier(uint32_t id, uint32_t count) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}

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

template <uint32_t S, bool kDyn>
__global__ void __launch_bounds__(kBlock)
gemm_fp8_bn128_kernel(const __grid_constant__ CUtensorMap tmap_a,
                      const __grid_constant__ CUtensorMap tmap_b,
                      const __grid_constant__ CUtensorMap tmap_d,
                      float* __restrict__ D, uint32_t M, uint32_t N,
                      uint32_t K, uint32_t group_m) {
  extern __shared__ __align__(1024) char smem_raw[];
  char* sA = smem_raw;                     // S x 16KB
  char* sB = smem_raw + S * kTileA;        // S x 16KB
  char* sEp = smem_raw + S * kStageBytes;  // 2 WG x 32KB
  uint64_t* s_go = reinterpret_cast<uint64_t*>(sEp + kEpBytes);  // 2（tile 交接）
  uint64_t* s_rel = s_go + 2;  // 2（tile 读释放，count 256）
  uint64_t* full = s_rel + 2;
  uint64_t* empty = full + S;
  uint32_t* s_tile = reinterpret_cast<uint32_t*>(empty + S);  // 2

  const uint32_t tid = threadIdx.x;
  const uint32_t num_bm = M / kBM, num_bn = N / kBN;
  const uint32_t total_tiles = num_bm * num_bn;
  const uint32_t kblocks = K / kBK;

  if (tid == 0) {
    for (uint32_t s = 0; s < S; ++s) {
      hopper::mbarrier_init(&full[s], 1);
      hopper::mbarrier_init(&empty[s], kConsumers);
    }
    if (kDyn) {
      hopper::mbarrier_init(&s_go[0], 1);
      hopper::mbarrier_init(&s_go[1], 1);
      hopper::mbarrier_init(&s_rel[0], kConsumers);
      hopper::mbarrier_init(&s_rel[1], kConsumers);
    }
  }
  __syncthreads();

  const uint32_t wg = (tid - 128) / 128;
  if (tid < 128) {
    // ---- producer: 全局连续 stage 环 + tile 获取（static 计算 / dynamic 队列）----
    if (tid == 0) {
      uint32_t j = 0, taken = 0;
      for (;;) {
        uint32_t tile;
        if (kDyn) {
          tile = atomicAdd(&g_tile_ctr, 1u);
          const uint32_t p = taken & 1;
          if (taken >= 2) {
            // 覆写/重翻转前等上轮读者读完——领先上锁到消费者读进度
            hopper::mbarrier_wait_parity(&s_rel[p], ((taken >> 1) - 1) & 1);
          }
          s_tile[p] = tile;
          hopper::mbarrier_arrive(&s_go[p]);  // 越界值也交接，消费者据此退出
          if (tile >= total_tiles) break;
        } else {
          tile = blockIdx.x + taken * gridDim.x;
          if (tile >= total_tiles) break;
        }
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
        ++taken;
      }
      if (kDyn) {
        // 自复位：最后一个退出的 producer 清队列（消费者的计算不碰计数器）
        if (atomicAdd(&g_done_ctr, 1u) == gridDim.x - 1) {
          g_tile_ctr = 0;
          g_done_ctr = 0;
        }
      }
    }
  } else {
    // ---- consumer: 每 tile 64 累加器（n128），环连续 ----
    uint32_t j = 0, done = 0;
    const uint32_t warp = (tid - 128) % 128 / 32, lane = tid % 32;
    for (;;) {
      uint32_t tile;
      if (kDyn) {
        const uint32_t p = done & 1;
        hopper::mbarrier_wait_parity(&s_go[p], (done >> 1) & 1);
        tile = s_tile[p];
        hopper::mbarrier_arrive(&s_rel[p]);  // 读完即释放；越界值也放行 producer
        if (tile >= total_tiles) break;
      } else {
        tile = blockIdx.x + done * gridDim.x;
        if (tile >= total_tiles) break;
      }
      uint32_t bm, bn;
      tile_to_bmn(tile, num_bm, num_bn, group_m, bm, bn);
      float d[64] = {};
      hopper::wgmma_fence();
      for (uint32_t kb = 0; kb < kblocks; ++kb, ++j) {
        hopper::mbarrier_wait_parity(&full[j % S], (j / S) & 1);
        const char* sa = sA + (j % S) * kTileA + wg * 8192;  // 本 WG 的 64 行半区
        const char* sb = sB + (j % S) * kTileB;              // 两 WG 共用全宽 B
#pragma unroll
        for (uint32_t kk = 0; kk < kBK / 32; ++kk) {
          const uint64_t da = hopper::gmma_desc_k_sw128(sa + kk * 32, 1024, 0);
          const uint64_t db = hopper::gmma_desc_k_sw128(sb + kk * 32, 1024, 0);
          hopper::wgmma_m64n128k32_f32_e4m3(da, db, d, (kb || kk) ? 1u : 0u);
        }
        hopper::wgmma_commit_group();
        if (kb >= 1) {
          hopper::wgmma_wait_group<1>();
          hopper::mbarrier_arrive(&empty[(j - 1) % S]);
        }
      }
      hopper::wgmma_wait_group<0>();
      hopper::mbarrier_arrive(&empty[(j - 1) % S]);  // 尾级补释放（persistent）

      // TMA store epilogue：4 个 {32,64} box / WG（n/32 ∈ [0,4)，布局同 5c/7b/8a）
      char* stage = sEp + wg * 32768;
#pragma unroll
      for (uint32_t r = 0; r < 64; r += 2) {
        const uint32_t m = 16 * warp + lane / 4 + 8 * ((r / 2) % 2);
        const uint32_t n = 2 * (lane % 4) + r % 2 + 8 * (r / 4);
        const uint32_t nl = n % 32;
        char* p = stage + (n / 32) * 8192 + m * 128 +
                  ((nl / 4) ^ (m % 8)) * 16 + (nl % 4) * 4;
        *reinterpret_cast<float2*>(p) = make_float2(d[r], d[r + 1]);
      }
      named_barrier(1 + wg, 128);
      hopper::fence_proxy_async_shared_cta();
      if ((tid - 128) % 128 == 0) {
        const int32_t row = int32_t(bm * kBM + wg * 64);
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          hopper::tma_store_2d(stage + i * 8192, &tmap_d,
                               int32_t(bn * kBN + i * 32), row);
        }
        hopper::tma_store_commit_group();
        hopper::tma_store_wait_read<0>();
      }
      ++done;
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

template <uint32_t S, bool kDyn>
void launch_config(const CUtensorMap& ta, const CUtensorMap& tb,
                   const CUtensorMap& td, float* d, uint32_t M, uint32_t N,
                   uint32_t K, uint32_t group_m) {
  const size_t smem = S * kStageBytes + kEpBytes + (4 + 2 * S) * sizeof(uint64_t) +
                      2 * sizeof(uint32_t);
  CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(gemm_fp8_bn128_kernel<S, kDyn>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int sms = 0, occ = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, gemm_fp8_bn128_kernel<S, kDyn>, kBlock, smem));
  const uint32_t total = (M / kBM) * (N / kBN);
  const uint32_t grid = std::min(total, uint32_t(sms * occ));
  gemm_fp8_bn128_kernel<S, kDyn><<<grid, kBlock, smem>>>(
      ta, tb, td, d, M, N, K, group_m);
  CUDA_CHECK_LAST();
}

void dispatch(uint32_t stages, bool dyn, const CUtensorMap& ta,
              const CUtensorMap& tb, const CUtensorMap& td, float* d,
              uint32_t M, uint32_t N, uint32_t K, uint32_t group_m) {
  if (group_m == 0) {
    std::fprintf(stderr, "group_m 须 >= 1\n");
    std::exit(1);
  }
  switch (stages) {
    case 2:
      dyn ? launch_config<2, true>(ta, tb, td, d, M, N, K, group_m)
          : launch_config<2, false>(ta, tb, td, d, M, N, K, group_m);
      break;
    case 3:
      dyn ? launch_config<3, true>(ta, tb, td, d, M, N, K, group_m)
          : launch_config<3, false>(ta, tb, td, d, M, N, K, group_m);
      break;
    case 4:
      dyn ? launch_config<4, true>(ta, tb, td, d, M, N, K, group_m)
          : launch_config<4, false>(ta, tb, td, d, M, N, K, group_m);
      break;
    case 5:
      dyn ? launch_config<5, true>(ta, tb, td, d, M, N, K, group_m)
          : launch_config<5, false>(ta, tb, td, d, M, N, K, group_m);
      break;
    default: std::fprintf(stderr, "未支持的 stage 数: %u\n", stages); std::exit(1);
  }
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  // ---- 正确性: 512 单波 + 2048 多波（覆盖 static/dynamic 两种获取）----
  {
    struct Shape { uint32_t m, n, k; const char* tag; };
    for (Shape sh : {Shape{512, 512, 256, "512"}, Shape{2048, 2048, 256, "2048 多波"}}) {
      const uint32_t M = sh.m, N = sh.n, K = sh.k;
      auto ha = hopper::make_random_vector(size_t(M) * K, 2001, -1.f, 1.f);
      auto hb = hopper::make_random_vector(size_t(N) * K, 2002, -1.f, 1.f);
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

      for (uint32_t s : {2u, 4u, 5u}) {
        for (uint32_t g : {1u, 8u}) {
          for (bool dyn : {false, true}) {
            bool ok = true;
            for (uint32_t rep = 0; rep < 3; ++rep) {
              dispatch(s, dyn, ta, tb, td, dC.get(), M, N, K, g);
              auto r = hopper::check_close(dC.download(), ref, 3e-2, 3e-2);
              ok = ok && r.pass;
              if (!r.pass) {
                char tag[80];
                std::snprintf(tag, sizeof(tag), "bn128 %s S=%u G=%u %s rep=%u",
                              sh.tag, s, g, dyn ? "dyn" : "sta", rep);
                hopper::print_report(r, tag);
              }
            }
            std::printf("[CHECK ] bn128 %s S=%u G=%-2u %s x3: %s\n", sh.tag, s, g,
                        dyn ? "dyn" : "sta", ok ? "PASS" : "FAIL");
            if (!ok) return 1;
          }
        }
      }
    }
  }

  // ---- 吞吐: S × G × sched 扫描 ----
  std::filesystem::create_directories("results");
  {
    constexpr uint32_t M = 2048, N = 2048, K = 1024;
    auto a = quantize(hopper::make_random_vector(size_t(M) * K, 2003, -1.f, 1.f));
    auto b = quantize(hopper::make_random_vector(size_t(N) * K, 2004, -1.f, 1.f));
    hopper::DeviceBuffer<__nv_fp8_e4m3> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_fp8(dA.get(), M, K, kBM);
    alignas(64) CUtensorMap tb = make_tmap_fp8(dB.get(), N, K, kBN);
    alignas(64) CUtensorMap td = make_tmap_f(dC.get(), M, N);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 2048x2048x1024 fp8 BN=128 persistent（8a 同日 200.3T / 7b 208.4T）\n");
    std::printf("  %-7s %-7s %-5s %12s %10s\n", "stages", "group_m", "sched", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_fp8_bn128_2048.csv",
                          {"stages", "group_m", "sched", "min_ms", "tflops"});
    for (uint32_t s : {2u, 3u, 4u, 5u}) {
      for (uint32_t g : {1u, 8u}) {
        for (bool dyn : {false, true}) {
          auto st = hopper::time_reps(3, 10, [&] {
            dispatch(s, dyn, ta, tb, td, dC.get(), M, N, K, g);
          });
          const double tf = hopper::tflops(flops, st.min_ms);
          std::printf("  %-7u %-7u %-5s %12.3f %10.2f\n", s, g, dyn ? "dyn" : "sta",
                      st.min_ms, tf);
          csv.row({std::to_string(s), std::to_string(g), dyn ? "dyn" : "sta",
                   std::to_string(st.min_ms), std::to_string(tf)});
        }
      }
    }
  }
  {
    constexpr uint32_t M = 8192, N = 8192, K = 8192;
    auto a = quantize(hopper::make_random_vector(size_t(M) * K, 2005, -1.f, 1.f));
    auto b = quantize(hopper::make_random_vector(size_t(N) * K, 2006, -1.f, 1.f));
    hopper::DeviceBuffer<__nv_fp8_e4m3> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_fp8(dA.get(), M, K, kBM);
    alignas(64) CUtensorMap tb = make_tmap_fp8(dB.get(), N, K, kBN);
    alignas(64) CUtensorMap td = make_tmap_f(dC.get(), M, N);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 8192x8192x8192 fp8 BN=128 persistent（8a/7b 同日 174.5T / 快窗 260T）\n");
    std::printf("  %-7s %-7s %-5s %12s %10s\n", "stages", "group_m", "sched", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_fp8_bn128_8192.csv",
                          {"stages", "group_m", "sched", "min_ms", "tflops"});
    for (uint32_t s : {3u, 4u, 5u}) {
      for (uint32_t g : {1u, 8u}) {
        for (bool dyn : {false, true}) {
          auto st = hopper::time_reps(3, 5, [&] {
            dispatch(s, dyn, ta, tb, td, dC.get(), M, N, K, g);
          });
          const double tf = hopper::tflops(flops, st.min_ms);
          std::printf("  %-7u %-7u %-5s %12.3f %10.2f\n", s, g, dyn ? "dyn" : "sta",
                      st.min_ms, tf);
          csv.row({std::to_string(s), std::to_string(g), dyn ? "dyn" : "sta",
                   std::to_string(st.min_ms), std::to_string(tf)});
        }
      }
    }
  }
  return 0;
}
