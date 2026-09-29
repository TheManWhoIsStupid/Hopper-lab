// Phase 4b: wgmma + SW128 swizzle 布局——量化 swizzle 对 wgmma 读 smem 的收益。
// 布局: K-major SW128，8x64(fp16) swizzle 原子（行 128B），原子沿 M 步进 SBO=1024B、
//       沿 K 方向再堆叠（K=128 = 2 个 k-group，k-group 步距 8KB）。
//       元素 (m,k) 字节偏移 = (k/64*8 + m/8)*1024 + (m%8)*128 + (((k%64/8)^(m%8))*16B + k%8*2B)
//       ——与 Phase 1b 验证过的 TMA SWIZZLE_128B 公式同源。
// 描述符: swizzle=B128(1), LBO 字段=1(固定), SBO=1024B, base_offset=0。
//   k16 步进(ks): start = (ks/4)*8KB + 32B*(ks%4)（行 0 的 swizzle XOR=0）。
//   CUTLASS DescriptorIterator 同样只推进 start_address 不改 base_offset——硬件按地址位恢复相位。
//   正确性部分额外扫描 base_offset 0..7，经验性确认该字段语义。
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

constexpr uint32_t kTM = 64, kTN = 64, kK = 128;
constexpr uint32_t kAtomElems = 64;  // SW128 原子宽 = 64 个 fp16 = 128B
constexpr uint32_t kKGroup = kK / kAtomElems;
constexpr uint32_t kAtomBytes = 1024;

// SW128 K-major 元素 (m,k) 的 smem 字节偏移（tile 起点需 1024B 对齐）
__device__ __forceinline__ uint32_t sw128_off(uint32_t m, uint32_t k) {
  const uint32_t atom = (k / kAtomElems) * (kTM / 8) + m / 8;
  const uint32_t c = k % kAtomElems;
  return atom * kAtomBytes + (m % 8) * 128 +
         ((((c / 8) ^ (m % 8)) * 8 + c % 8) * 2);
}

__global__ void __launch_bounds__(128)
wgmma_sw128_kernel(const __half* __restrict__ A, const __half* __restrict__ Bt,
                   float* __restrict__ D, uint32_t M, uint32_t N, uint32_t K,
                   uint32_t base_off) {
  const uint32_t tid = threadIdx.x;
  const uint32_t bm = blockIdx.y, bn = blockIdx.x;

  __shared__ __align__(1024) __half sA[kTM * kK];  // 16KB
  __shared__ __align__(1024) __half sB[kTN * kK];  // 16KB

  // gmem -> smem：每线程搬若干 16B 行块（8 元素）
  constexpr uint32_t kChunks = kTM * (kK / 8);  // 64*16 = 1024 块
  const char* ga = reinterpret_cast<const char*>(A + size_t(bm * kTM) * K);
  const char* gb = reinterpret_cast<const char*>(Bt + size_t(bn * kTN) * K);
  for (uint32_t idx = tid; idx < kChunks; idx += 128) {
    const uint32_t m = idx / (kK / 8), j = idx % (kK / 8);  // j: 16B 块号(跨 k-group)
    const uint32_t k = j * 8;
    *reinterpret_cast<uint4*>(reinterpret_cast<char*>(sA) + sw128_off(m, k)) =
        *reinterpret_cast<const uint4*>(ga + size_t(m) * K * 2 + k * 2);
    *reinterpret_cast<uint4*>(reinterpret_cast<char*>(sB) + sw128_off(m, k)) =
        *reinterpret_cast<const uint4*>(gb + size_t(m) * K * 2 + k * 2);
  }
  __syncthreads();
  hopper::fence_proxy_async_shared_cta();

  float d[32] = {};
  hopper::wgmma_fence();
  for (uint32_t ks = 0; ks < kK / 16; ++ks) {
    const uint32_t kbyte = (ks / 4) * (kTM / 8) * kAtomBytes + (ks % 4) * 32;
    const uint64_t da = hopper::gmma_desc_k_sw128(
        reinterpret_cast<const char*>(sA) + kbyte, kAtomBytes, base_off);
    const uint64_t db = hopper::gmma_desc_k_sw128(
        reinterpret_cast<const char*>(sB) + kbyte, kAtomBytes, base_off);
    hopper::wgmma_m64n64k16_f32_f16(da, db, d, ks > 0 ? 1u : 0u);
  }
  hopper::wgmma_commit_group();
  hopper::wgmma_wait_group_0();

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

  constexpr uint32_t M = 512, N = 512, K = kK;
  auto ha = hopper::make_random_vector(size_t(M) * K, 301, -1.f, 1.f);
  auto hb = hopper::make_random_vector(size_t(N) * K, 302, -1.f, 1.f);
  std::vector<__half> a(ha.begin(), ha.end()), b(hb.begin(), hb.end());
  hopper::DeviceBuffer<__half> dA(a.size()), dB(b.size());
  hopper::DeviceBuffer<float> dC(size_t(M) * N);
  dA.upload(a);
  dB.upload(b);

  std::vector<float> ref(size_t(M) * N);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n) {
      float acc = 0.f;
      for (uint32_t k = 0; k < K; ++k)
        acc += float(a[size_t(m) * K + k]) * float(b[size_t(n) * K + k]);
      ref[size_t(m) * N + n] = acc;
    }

  // ---- base_offset 经验扫描 ----
  uint32_t good_base = 0xFFFFFFFF;
  std::printf("\n[SWEEP  ] base_offset 语义扫描（0..7）\n");
  for (uint32_t base = 0; base < 8; ++base) {
    wgmma_sw128_kernel<<<dim3(N / kTN, M / kTM), 128>>>(dA.get(), dB.get(),
                                                        dC.get(), M, N, K, base);
    CUDA_CHECK_LAST();
    auto rep = hopper::check_close(dC.download(), ref, 1e-3, 1e-3);
    std::printf("  base_offset=%u : %s (max_abs=%.3e)\n", base,
                rep.pass ? "PASS" : "fail", rep.max_abs_err);
    if (rep.pass && good_base == 0xFFFFFFFF) good_base = base;
  }
  if (good_base == 0xFFFFFFFF) {
    std::printf("  所有 base_offset 均失败！布局/描述符有误，暂停\n");
    return 1;
  }

  // ---- 吞吐: 2048x2048x128 ----
  {
    constexpr uint32_t MB = 2048, NB = 2048;
    auto ha2 = hopper::make_random_vector(size_t(MB) * K, 303, -1.f, 1.f);
    auto hb2 = hopper::make_random_vector(size_t(NB) * K, 304, -1.f, 1.f);
    std::vector<__half> a2(ha2.begin(), ha2.end()), b2(hb2.begin(), hb2.end());
    hopper::DeviceBuffer<__half> dA2(a2.size()), dB2(b2.size());
    hopper::DeviceBuffer<float> dC2(size_t(MB) * NB);
    dA2.upload(a2);
    dB2.upload(b2);

    dim3 grid(NB / kTN, MB / kTM);
    const double flops = 2.0 * MB * NB * K;
    auto st = hopper::time_reps(3, 10, [&] {
      wgmma_sw128_kernel<<<grid, 128>>>(dA2.get(), dB2.get(), dC2.get(), MB, NB,
                                        K, good_base);
      CUDA_CHECK_LAST();
    });
    std::printf("\n[WGMMA ] 2048x2048x128 SW128 (base=%u): min %.3f ms -> %.2f TFLOPS\n",
                good_base, st.min_ms, hopper::tflops(flops, st.min_ms));

    std::filesystem::create_directories("results");
    hopper::CsvWriter csv("results/wgmma_sw128.csv", {"case", "min_ms", "tflops"});
    csv.row({"bench 2048x2048x128 sw128",
             std::to_string(st.min_ms), std::to_string(hopper::tflops(flops, st.min_ms))});
  }
  return 0;
}
