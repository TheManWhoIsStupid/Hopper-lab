// Phase 5: 综合 GEMM —— TMA 多级流水线 + wgmma 异步重叠（Phase 1b + 2 + 4 的拼装）。
// C = A × B: A (MxK row-major fp16), Bt (NxK row-major fp16, K-major B), C fp32 (MxN)
// tile: BM=BN=BK=64。每 stage: A 8KB + B 8KB，TMA SWIZZLE_128B 直接按 wgmma
//       SW128 规范布局写入（Phase 1b 公式 = Phase 4b 公式，端到端闭环）。
// block = 128 线程 = 1 warpgroup；tid0 兼 producer（TMA 发射），全体消费（wgmma）。
//
// 流水线时序（S>=2）:
//   prologue: tid0 发 j=0..S-2 的 TMA（首轮 stage 天然空闲，不等 empty）
//   迭代 i:  wait full[i%S] -> 4 拍 wgmma k16 -> commit
//             -> wait_group<1>（上组读完 smem）-> 全体 arrive empty[(i-1)%S]
//             -> tid0 issue(i+S-1)（empty[(i-1)%S] 刚释放，立即复用装载数据）
//   wgmma 保持 1 组在飞跨越迭代边界，TMA 领先 S-1 个 stage——双异步重叠。
//   注意 issue 放在迭代末尾（Phase 2 是开头）：wgmma 的 smem 读取是异步的，
//   释放必须滞后一拍，issue 必须在释放之后，否则 tid0 自锁。
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

constexpr uint32_t kBM = 64, kBN = 64, kBK = 64;
constexpr uint32_t kTileBytes = kBM * kBK * 2;   // 8KB (fp16)
constexpr uint32_t kStageBytes = 2 * kTileBytes; // A + B = 16KB
constexpr uint32_t kBlock = 128;                 // 一个 warpgroup

template <uint32_t S>
__global__ void __launch_bounds__(kBlock)
gemm_fused_kernel(const __grid_constant__ CUtensorMap tmap_a,
                  const __grid_constant__ CUtensorMap tmap_b,
                  float* __restrict__ D, uint32_t M, uint32_t N, uint32_t K) {
  extern __shared__ __align__(1024) char smem_raw[];
  char* sA = smem_raw;                       // S x 8KB，SW128 原子需 1024B 对齐
  char* sB = smem_raw + S * kTileBytes;      // S x 8KB
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_raw + 2 * S * kTileBytes);
  uint64_t* empty = full + S;

  const uint32_t tid = threadIdx.x;
  const uint32_t bm = blockIdx.y, bn = blockIdx.x;
  const uint32_t kblocks = K / kBK;

  if (tid == 0) {
    for (uint32_t s = 0; s < S; ++s) {
      hopper::mbarrier_init(&full[s], 1);      // TMA 记账型
      hopper::mbarrier_init(&empty[s], kBlock); // 全体消费 arrive
    }
  }
  __syncthreads();

  auto issue = [&](uint32_t j) {
    if (j >= S) {  // 首轮 S 个 stage 天然空闲
      hopper::mbarrier_wait_parity(&empty[j % S], ((j / S) - 1) & 1);
    }
    hopper::mbarrier_arrive_expect_tx(&full[j % S], kStageBytes);
    hopper::tma_load_2d(sA + (j % S) * kTileBytes, &tmap_a, int32_t(j * kBK),
                        int32_t(bm * kBM), &full[j % S]);
    hopper::tma_load_2d(sB + (j % S) * kTileBytes, &tmap_b, int32_t(j * kBK),
                        int32_t(bn * kBN), &full[j % S]);
  };

  // prologue: 填 S-1 个 load（S>=2 是本内核的不变式；无流水线基线见 Phase 4）
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
      // stage 内 k16 步进 = 32B（BK=64 恰好一个 SW128 原子宽，无跨原子相位问题）
      const uint64_t da = hopper::gmma_desc_k_sw128(sa + kk * 32, 1024, 0);
      const uint64_t db = hopper::gmma_desc_k_sw128(sb + kk * 32, 1024, 0);
      hopper::wgmma_m64n64k16_f32_f16(da, db, d, (i || kk) ? 1u : 0u);
    }
    hopper::wgmma_commit_group();
    if (i >= 1) {
      hopper::wgmma_wait_group<1>();  // 上一组已读完其 stage
      hopper::mbarrier_arrive(&empty[(i - 1) % S]);
    }
    if (tid == 0 && i + S - 1 < kblocks) {
      issue(i + S - 1);  // empty[(i-1)%S] 刚释放，安全复用
    }
  }
  hopper::wgmma_wait_group<0>();

  // 累加器写出（映射同 Phase 4）
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
  const cuuint64_t gdim[2] = {cols, rows};             // dim0 = 内维(列)
  const cuuint64_t gstride[1] = {cols * sizeof(__half)}; // 行步长(字节)
  const cuuint32_t box[2] = {64, 64};                  // 内维 64 fp16 = 128B(SW128)
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
  const size_t smem = 2 * S * kTileBytes + 2 * S * sizeof(uint64_t);
  CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(gemm_fused_kernel<S>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  gemm_fused_kernel<S><<<dim3(N / kBN, M / kBM), kBlock, smem>>>(ta, tb, d, M, N,
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
    auto ha = hopper::make_random_vector(size_t(M) * K, 401, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 402, -1.f, 1.f);
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
      dispatch(s, ta, tb, dC.get(), M, N, K);
      auto rep = hopper::check_close(dC.download(), ref, 1e-3, 1e-3);
      char tag[32];
      std::snprintf(tag, sizeof(tag), "fused S=%u 512^2x256", s);
      hopper::print_report(rep, tag);
      if (!rep.pass) return 1;
    }
  }

  // ---- 吞吐: 2048x2048x1024, S 扫描 ----
  {
    constexpr uint32_t M = 2048, N = 2048, K = 1024;
    auto ha = hopper::make_random_vector(size_t(M) * K, 403, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 404, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_h(dA.get(), M, K);
    alignas(64) CUtensorMap tb = make_tmap_h(dB.get(), N, K);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 2048x2048x1024 TMA+wgmma 流水线 stage 扫描\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");

    std::filesystem::create_directories("results");
    hopper::CsvWriter csv("results/gemm_fused.csv", {"stages", "min_ms", "tflops"});
    for (uint32_t s : {2u, 3u, 4u, 6u, 8u}) {
      auto st = hopper::time_reps(3, 10, [&] {
        dispatch(s, ta, tb, dC.get(), M, N, K);
      });
      const double tf = hopper::tflops(flops, st.min_ms);
      std::printf("  %-7u %12.3f %10.2f\n", s, st.min_ms, tf);
      csv.row({std::to_string(s), std::to_string(st.min_ms), std::to_string(tf)});
    }
  }

  // ---- 大尺寸: 8192^3, 与 cuBLAS HGEMM（Phase 3 contended 72.59T）同尺寸对比 ----
  {
    constexpr uint32_t M = 8192, N = 8192, K = 8192;
    auto ha = hopper::make_random_vector(size_t(M) * K, 405, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 406, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);
    alignas(64) CUtensorMap ta = make_tmap_h(dA.get(), M, K);
    alignas(64) CUtensorMap tb = make_tmap_h(dB.get(), N, K);

    const double flops = 2.0 * M * N * K;
    std::printf("\n[GEMM  ] 8192x8192x8192（cuBLAS contended 基线 72.59T）\n");
    std::printf("  %-7s %12s %10s\n", "stages", "min_ms", "TFLOPS");
    hopper::CsvWriter csv("results/gemm_fused_8192.csv", {"stages", "min_ms", "tflops"});
    for (uint32_t s : {2u, 3u, 4u, 6u, 8u}) {
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
