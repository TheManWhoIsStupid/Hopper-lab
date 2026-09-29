// Phase 4a: wgmma m64n64k16 首战——正确性 + 吞吐基线。
// 结构：每 block 一个 warpgroup(128 线程) 算一个 64x64 输出 tile，K=128 整段驻留 smem，
//       K 方向 8 拍 wgmma（scale_d: 首拍覆盖、后续累加）。
// 布局：K-major INTERLEAVE（无 swizzle）—— 8x8 核心矩阵连续存放 128B。
//       gmem -> smem 按核心矩阵搬运（每行 16B = uint4），描述符 LBO=128B, SBO=16*K。
// 对比：Phase 4b 换 SW128 布局量化 swizzle 对 wgmma 读 smem 的影响；流水线化留给 Phase 6。
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "common/bench.h"
#include "common/checker.h"
#include "common/device_info.h"
#include "common/errors.h"
#include "common/gmma.cuh"
#include "common/tensor.h"
#include "common/timer.h"

namespace {

constexpr uint32_t kTM = 64;   // wgmma M
constexpr uint32_t kTN = 64;   // wgmma N
constexpr uint32_t kK = 128;   // K 深度（整段驻留 smem，单 stage）
constexpr uint32_t kKCore = kK / 8;
constexpr uint32_t kCores = (kTM / 8) * kKCore;  // 每侧核心矩阵数 = 8*16 = 128

// 一个 block = 一个 warpgroup = 一个 64x64 tile
// A: MxK row-major；Bt: NxK row-major（K-major B，即 B 的转置存储）；D: MxN fp32 row-major
__global__ void __launch_bounds__(128)
wgmma_basic_kernel(const __half* __restrict__ A, const __half* __restrict__ Bt,
                   float* __restrict__ D, uint32_t M, uint32_t N, uint32_t K) {
  const uint32_t tid = threadIdx.x;
  const uint32_t bm = blockIdx.y, bn = blockIdx.x;

  __shared__ __half sA[kTM * kK];  // 16KB
  __shared__ __half sB[kTN * kK];  // 16KB

  // gmem -> smem：thread 各搬若干核心矩阵（8 行 x 16B/行）
  for (uint32_t cm = tid; cm < kCores; cm += 128) {
    const uint32_t i = cm / kKCore, j = cm % kKCore;  // i: 8行组, j: k方向核心矩阵
#pragma unroll
    for (uint32_t r = 0; r < 8; ++r) {
      const uint32_t m = i * 8 + r, k = j * 8;
      *reinterpret_cast<uint4*>(&sA[cm * 64 + r * 8]) =
          *reinterpret_cast<const uint4*>(A + size_t(bm * kTM + m) * K + k);
      *reinterpret_cast<uint4*>(&sB[cm * 64 + r * 8]) =
          *reinterpret_cast<const uint4*>(Bt + size_t(bn * kTN + m) * K + k);
    }
  }
  __syncthreads();
  hopper::fence_proxy_async_shared_cta();  // 通用 proxy 写对 wgmma(异步 proxy) 可见

  float d[32] = {};  // m64n64k16 f32 累加器 = 每线程 32 个
  hopper::wgmma_fence();
  for (uint32_t ks = 0; ks < kK / 16; ++ks) {
    // k16 步进 = 2 个核心矩阵 = 256B = 128 个 half
    const uint64_t da = hopper::gmma_desc_k_inter(sA + ks * 128, 128, kKCore * 128);
    const uint64_t db = hopper::gmma_desc_k_inter(sB + ks * 128, 128, kKCore * 128);
    hopper::wgmma_m64n64k16_f32_f16(da, db, d, ks > 0 ? 1u : 0u);
  }
  hopper::wgmma_commit_group();
  hopper::wgmma_wait_group_0();

  // 累加器写出：m = 16*warp + lane/4 + 8*((r/2)%2);  n = 2*(lane%4) + r%2 + 8*(r/4)
  const uint32_t warp = tid / 32, lane = tid % 32;
#pragma unroll
  for (uint32_t r = 0; r < 32; ++r) {
    const uint32_t m = 16 * warp + lane / 4 + 8 * ((r / 2) % 2);
    const uint32_t n = 2 * (lane % 4) + r % 2 + 8 * (r / 4);
    D[size_t(bm * kTM + m) * N + bn * kTN + n] = d[r];
  }
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);
  static_assert(kK == 128, "本实验 K 固定 128（单 stage smem 驻留）");

  // ---- 正确性: 512x512x128 vs CPU ----
  {
    constexpr uint32_t M = 512, N = 512, K = kK;
    auto ha = hopper::make_random_vector(size_t(M) * K, 201, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 202, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);

    dim3 grid(N / kTN, M / kTM);
    wgmma_basic_kernel<<<grid, 128>>>(dA.get(), dB.get(), dC.get(), M, N, K);
    CUDA_CHECK_LAST();

    auto got = dC.download();
    std::vector<float> ref(size_t(M) * N);
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t n = 0; n < N; ++n) {
        float acc = 0.f;
        for (uint32_t k = 0; k < K; ++k)
          acc += float(a[size_t(m) * K + k]) * float(b[size_t(n) * K + k]);
        ref[size_t(m) * N + n] = acc;
      }
    auto rep = hopper::check_close(got, ref, 1e-3, 1e-3);
    hopper::print_report(rep, "wgmma 512x512x128");
    if (!rep.pass) return 1;
  }

  // ---- 吞吐: 2048x2048x128 ----
  {
    constexpr uint32_t M = 2048, N = 2048, K = kK;
    auto ha = hopper::make_random_vector(size_t(M) * K, 203, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 204, -1.f, 1.f);
    std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
    hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);

    dim3 grid(N / kTN, M / kTM);  // 32x32 = 1024 blocks
    const double flops = 2.0 * M * N * K;
    auto st = hopper::time_reps(3, 10, [&] {
      wgmma_basic_kernel<<<grid, 128>>>(dA.get(), dB.get(), dC.get(), M, N, K);
      CUDA_CHECK_LAST();
    });
    std::printf("\n[WGMMA ] 2048x2048x128 no-swizzle INTERLEAVE: min %.3f ms -> %.2f TFLOPS\n",
                st.min_ms, hopper::tflops(flops, st.min_ms));

    std::filesystem::create_directories("results");
    hopper::CsvWriter csv("results/wgmma_basic.csv", {"case", "min_ms", "tflops"});
    csv.row({"bench 2048x2048x128 interleaved", std::to_string(st.min_ms),
             std::to_string(hopper::tflops(flops, st.min_ms))});
  }
  return 0;
}
