// Phase 7b: fp8 (e4m3) warp-specialized GEMM —— 5b/5c 结构 × fp8 数据通路。
//
//   与 fp16 版（gemm_ws.cu）的全部差异：
//   * 输入 e4m3（1B），kBK 64→128（fp8 的 SW128 原子宽 = 128 元素 = 128B，
//     TMA box 内维必须 128B 满）；stage 字节数不变：A {128,128}=16KB、B {128,64}=8KB
//   * 每 kblock 4 条 wgmma m64n64k32（fp16 是 4 条 k16）——原子内 k 步进
//     仍 32B、consumer 半区偏移仍 8192、SBO 仍 1024，描述符调用原封不动
//     （SW128 原子按字节定义，与元素位宽无关；7a 已实证）
//   * tmap dataType UINT8（TMA 只管搬字节），gstride = cols * 1
//   * D 仍 fp32，epilogue（plain / TMA store）零改动
//
//   数值：fp8 指令内部累加 ~fp22（7a 实证：单指令残差 7.6e-4，随指令数
//   线性累积，K=256 时 ~1.5e-2）——容差 3e-2，判别面（布局错 ~35）差 3 个量级。
//
//   对照（fp16 ws，满载）：2048³×1024 117.6T / 8192³ 73.1T；快窗 8192³ 130.1T。
//   fp8 算力峰值 = 2×fp16（296T）——快窗若 compute-bound 应见 ~2×。
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
constexpr uint32_t kStageBytes = kTileA + kTileB;  // 24KB，与 fp16 版相同
constexpr uint32_t kBlock = 384;                   // 1 producer + 2 consumer WG
constexpr uint32_t kConsumers = 256;

__device__ __forceinline__ void named_barrier(uint32_t id, uint32_t count) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}

template <uint32_t S, bool kTmaStore>
__global__ void __launch_bounds__(kBlock)
gemm_fp8_kernel(const __grid_constant__ CUtensorMap tmap_a,
                const __grid_constant__ CUtensorMap tmap_b,
                const __grid_constant__ CUtensorMap tmap_d,
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
    return;
  }

  // ---- consumer warpgroups: 各管 64 行半区，每 kblock 4 条 k32 ----
  const uint32_t wg = (tid - 128) / 128;
  float d[32] = {};
  hopper::wgmma_fence();
  for (uint32_t i = 0; i < kblocks; ++i) {
    hopper::mbarrier_wait_parity(&full[i % S], (i / S) & 1);
    const char* sa = sA + (i % S) * kTileA + wg * 8192;  // 本 WG 的 64 行半区
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
      hopper::mbarrier_arrive(&empty[(i - 1) % S]);
    }
  }
  hopper::wgmma_wait_group<0>();

  const uint32_t warp = (tid - 128) % 128 / 32, lane = tid % 32;
  if (kTmaStore) {
    // TMA store epilogue：与 fp16 版逐字节相同（D 是 fp32，暂存/box 不变）
    named_barrier(3, 256);  // 两个 consumer WG 的 wgmma 全部排空（5c 的教训）
    char* stage = sA + wg * 16384;
#pragma unroll
    for (uint32_t r = 0; r < 32; r += 2) {
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
      hopper::tma_store_2d(stage, &tmap_d, int32_t(bn * kBN), row);
      hopper::tma_store_2d(stage + 8192, &tmap_d, int32_t(bn * kBN + 32), row);
      hopper::tma_store_commit_group();
      hopper::tma_store_wait_read<0>();
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

// fp8 输入的 tmap：dataType UINT8（TMA 按字节搬运），box 内维 = 128 元素 = 128B
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

template <uint32_t S, bool kTmaStore>
void launch_config(const CUtensorMap& ta, const CUtensorMap& tb,
                   const CUtensorMap& td, float* d, uint32_t M, uint32_t N,
                   uint32_t K) {
  const size_t smem = S * kStageBytes + 2 * S * sizeof(uint64_t);
  CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(gemm_fp8_kernel<S, kTmaStore>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  gemm_fp8_kernel<S, kTmaStore><<<dim3(N / kBN, M / kBM), kBlock, smem>>>(
      ta, tb, td, d, M, N, K);
  CUDA_CHECK_LAST();
}

void dispatch(uint32_t stages, bool tma, const CUtensorMap& ta,
              const CUtensorMap& tb, const CUtensorMap& td, float* d,
              uint32_t M, uint32_t N, uint32_t K) {
  switch (stages) {
    case 2:
      if (tma) launch_config<2, true>(ta, tb, td, d, M, N, K);
      else launch_config<2, false>(ta, tb, td, d, M, N, K);
      break;
    case 3:
      if (tma) launch_config<3, true>(ta, tb, td, d, M, N, K);
      else launch_config<3, false>(ta, tb, td, d, M, N, K);
      break;
    case 4:
      if (tma) launch_config<4, true>(ta, tb, td, d, M, N, K);
      else launch_config<4, false>(ta, tb, td, d, M, N, K);
      break;
    case 6:
      if (tma) launch_config<6, true>(ta, tb, td, d, M, N, K);
      else launch_config<6, false>(ta, tb, td, d, M, N, K);
      break;
    default: std::fprintf(stderr, "未支持的 stage 数: %u\n", stages); std::exit(1);
  }
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  // ---- 正确性: 512x512x256（K=256 = 2 个 kblock），S x epilogue x3 ----
  {
    constexpr uint32_t M = 512, N = 512, K = 256;
    auto ha = hopper::make_random_vector(size_t(M) * K, 901, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 902, -1.f, 1.f);
    auto a = quantize(ha), b = quantize(hb);
    hopper::DeviceBuffer<__nv_fp8_e4m3> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_fp8(dA.get(), M, K, kBM);
    alignas(64) CUtensorMap tb = make_tmap_fp8(dB.get(), N, K, kBN);
    alignas(64) CUtensorMap td = make_tmap_f(dC.get(), M, N);

    std::vector<float> ref(size_t(M) * N);  // 从反量化的同一份字节精确重算
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t n = 0; n < N; ++n) {
        float acc = 0.f;
        for (uint32_t k = 0; k < K; ++k)
          acc += float(a[size_t(m) * K + k]) * float(b[size_t(n) * K + k]);
        ref[size_t(m) * N + n] = acc;
      }

    for (bool tma : {false, true}) {
      for (uint32_t s : {2u, 3u, 4u}) {
        bool ok = true;
        for (uint32_t rep = 0; rep < 3; ++rep) {
          dispatch(s, tma, ta, tb, td, dC.get(), M, N, K);
          auto r = hopper::check_close(dC.download(), ref, 3e-2, 3e-2);
          ok = ok && r.pass;
          if (!r.pass) {
            char tag[48];
            std::snprintf(tag, sizeof(tag), "fp8 S=%u tma_ep=%d rep=%u", s,
                          (int)tma, rep);
            hopper::print_report(r, tag);
          }
        }
        std::printf("[CHECK ] fp8 S=%u tma_ep=%d x3        : %s\n", s, (int)tma,
                    ok ? "PASS" : "FAIL");
        if (!ok) return 1;
      }
    }
  }

  // ---- 吞吐: 对照 fp16 ws（满载 117.6T / 73.1T，快窗 130.1T） ----
  std::filesystem::create_directories("results");
  {
    constexpr uint32_t M = 2048, N = 2048, K = 1024;
    auto a = quantize(hopper::make_random_vector(size_t(M) * K, 903, -1.f, 1.f));
    auto b = quantize(hopper::make_random_vector(size_t(N) * K, 904, -1.f, 1.f));
    hopper::DeviceBuffer<__nv_fp8_e4m3> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_fp8(dA.get(), M, K, kBM);
    alignas(64) CUtensorMap tb = make_tmap_fp8(dB.get(), N, K, kBN);
    alignas(64) CUtensorMap td = make_tmap_f(dC.get(), M, N);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 2048x2048x1024 fp8 ws BM=128（fp16 对照 117.6T）\n");
    std::printf("  %-9s %-7s %12s %10s\n", "epilogue", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_fp8_2048.csv",
                          {"epilogue", "stages", "min_ms", "tflops"});
    for (bool tma : {false, true}) {
      for (uint32_t s : {2u, 3u, 4u, 6u}) {
        auto st = hopper::time_reps(3, 10, [&] {
          dispatch(s, tma, ta, tb, td, dC.get(), M, N, K);
        });
        const double tf = hopper::tflops(flops, st.min_ms);
        std::printf("  %-9s %-7u %12.3f %10.2f\n", tma ? "tma" : "plain", s,
                    st.min_ms, tf);
        csv.row({tma ? "tma" : "plain", std::to_string(s), std::to_string(st.min_ms),
                 std::to_string(tf)});
      }
    }
  }
  {
    constexpr uint32_t M = 8192, N = 8192, K = 8192;
    auto a = quantize(hopper::make_random_vector(size_t(M) * K, 905, -1.f, 1.f));
    auto b = quantize(hopper::make_random_vector(size_t(N) * K, 906, -1.f, 1.f));
    hopper::DeviceBuffer<__nv_fp8_e4m3> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_fp8(dA.get(), M, K, kBM);
    alignas(64) CUtensorMap tb = make_tmap_fp8(dB.get(), N, K, kBN);
    alignas(64) CUtensorMap td = make_tmap_f(dC.get(), M, N);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 8192x8192x8192 fp8 ws BM=128（fp16 满载 73.1T / 快窗 130.1T）\n");
    std::printf("  %-9s %-7s %12s %10s\n", "epilogue", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_fp8_8192.csv",
                          {"epilogue", "stages", "min_ms", "tflops"});
    for (bool tma : {false, true}) {
      for (uint32_t s : {2u, 3u, 4u}) {
        auto st = hopper::time_reps(3, 5, [&] {
          dispatch(s, tma, ta, tb, td, dC.get(), M, N, K);
        });
        const double tf = hopper::tflops(flops, st.min_ms);
        std::printf("  %-9s %-7u %12.3f %10.2f\n", tma ? "tma" : "plain", s,
                    st.min_ms, tf);
        csv.row({tma ? "tma" : "plain", std::to_string(s), std::to_string(st.min_ms),
                 std::to_string(tf)});
      }
    }
  }
  return 0;
}
