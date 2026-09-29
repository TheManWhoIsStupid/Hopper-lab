// Phase 7a: fp8 (e4m3) wgmma m64n64k32 正确性——指令封装 + 描述符复用假设的验证。
//
// 核心假设（对着 CUTLASS 源码推导，本文件实证）：
//   * SW128 原子按字节定义（Swizzle<3,4,3> ∘ 8 行 × 128B），fp16/fp8 同构；
//     fp8 的 SW128 原子宽 = 128 个元素，原子内 k32 核心矩阵步进 32B（每原子 4 步），
//     k-group（= 1 个原子宽 = 128 元素）沿 M 步距 SBO = 1024B——描述符函数不变。
//   * smem 元素 (m,k) 字节偏移（K-major SW128, fp8）：
//       (k/128 * TM/8 + m/8)*1024 + (m%8)*128 + ((k%128/16 ^ m%8)*16 + k%16)
//     与 Phase 1b/4b 的 fp16 公式同源（16B 块 = 16 个 fp8）。
//   * 累加器映射、base_off 语义与 fp16 全同。
//
// 正确性: 512x512x256（2 个 k-group），base_offset 0..7 扫描（沿袭 4b）。
// 吞吐:   2048x2048x{128,512}，K=128 直接对照 4b fp16 SW128（35.2T 同形状）。
#include <cuda_fp8.h>

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

constexpr uint32_t kTM = 64, kTN = 64;
constexpr uint32_t kAtomElems = 128;  // SW128 原子宽 = 128 个 fp8 = 128B
constexpr uint32_t kAtomBytes = 1024;

// SW128 K-major fp8 元素 (m,k) 的 smem 字节偏移（tile 起点需 1024B 对齐）
__device__ __forceinline__ uint32_t sw128_off_fp8(uint32_t m, uint32_t k) {
  const uint32_t atom = (k / kAtomElems) * (kTM / 8) + m / 8;
  const uint32_t c = k % kAtomElems;
  return atom * kAtomBytes + (m % 8) * 128 +
         ((((c / 16) ^ (m % 8)) * 16 + c % 16));
}

__global__ void __launch_bounds__(128)
wgmma_fp8_kernel(const __nv_fp8_e4m3* __restrict__ A,
                 const __nv_fp8_e4m3* __restrict__ Bt, float* __restrict__ D,
                 uint32_t M, uint32_t N, uint32_t K, uint32_t base_off,
                 uint32_t steps) {
  extern __shared__ __align__(1024) char smem_raw[];
  char* sA = smem_raw;                       // TM x K x 1B
  char* sB = smem_raw + kTM * K;             // TN x K x 1B

  const uint32_t tid = threadIdx.x;
  const uint32_t bm = blockIdx.y, bn = blockIdx.x;

  // gmem -> smem：每线程搬 16B 块（16 个 fp8），手工按 swizzle 公式落位
  const uint32_t chunks = kTM * (K / 16);
  const char* ga = reinterpret_cast<const char*>(A + size_t(bm * kTM) * K);
  const char* gb = reinterpret_cast<const char*>(Bt + size_t(bn * kTN) * K);
  for (uint32_t idx = tid; idx < chunks; idx += 128) {
    const uint32_t m = idx / (K / 16), j = idx % (K / 16);
    const uint32_t k = j * 16;
    *reinterpret_cast<uint4*>(sA + sw128_off_fp8(m, k)) =
        *reinterpret_cast<const uint4*>(ga + size_t(m) * K + k);
    *reinterpret_cast<uint4*>(sB + sw128_off_fp8(m, k)) =
        *reinterpret_cast<const uint4*>(gb + size_t(m) * K + k);
  }
  __syncthreads();
  hopper::fence_proxy_async_shared_cta();

  // steps: 实际发射的 k32 wgmma 条数（steps=1 是单指令精度探针，
  // 对应参考值只算前 32 列）。tile 布局始终是完整的 K 宽（SW128 要求原子满宽）
  float d[32] = {};
  hopper::wgmma_fence();
  for (uint32_t ks = 0; ks < steps; ++ks) {
    // k-group（128 元素）沿 M 堆叠；组内 4 步 k32，各 32B
    const uint32_t kbyte = (ks / 4) * (kTM / 8) * kAtomBytes + (ks % 4) * 32;
    const uint64_t da =
        hopper::gmma_desc_k_sw128(sA + kbyte, kAtomBytes, base_off);
    const uint64_t db =
        hopper::gmma_desc_k_sw128(sB + kbyte, kAtomBytes, base_off);
    hopper::wgmma_m64n64k32_f32_e4m3(da, db, d, ks > 0 ? 1u : 0u);
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

void launch(const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, float* d,
            uint32_t M, uint32_t N, uint32_t K, uint32_t base_off,
            uint32_t steps) {
  const size_t smem = 2 * size_t(kTM) * K;
  CUDA_CHECK(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(wgmma_fp8_kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  wgmma_fp8_kernel<<<dim3(N / kTN, M / kTM), 128, smem>>>(a, b, d, M, N, K,
                                                          base_off, steps);
  CUDA_CHECK_LAST();
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  // ---- 正确性: 512x512x256（2 k-group），base_offset 扫描 ----
  // 容差说明：e4m3 乘积在 fp32 内精确，参考值也从同一份 fp8 字节反量化重算，
  // 剩余残差来自 wgmma fp8 指令内部低于 fp32 的累加精度（见下方 K=32 探针），
  // 实测 ~1.5e-2（K=256）。布局错误的指纹是 ~35——差 3 个数量级，判别面足够。
  uint32_t good_base = 0xFFFFFFFF;
  {
    constexpr uint32_t M = 512, N = 512, K = 256;
    auto ha = hopper::make_random_vector(size_t(M) * K, 801, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(N) * K, 802, -1.f, 1.f);
    std::vector<__nv_fp8_e4m3> a, b;  // fp8 量化（RN/SATFINITE，[-1,1] 内无饱和）
    a.reserve(ha.size());
    b.reserve(hb.size());
    for (float f : ha) a.push_back(__nv_fp8_e4m3(f));
    for (float f : hb) b.push_back(__nv_fp8_e4m3(f));
    hopper::DeviceBuffer<__nv_fp8_e4m3> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(M) * N);
    dA.upload(a);
    dB.upload(b);

    // 参考：从反量化的 fp8 精确重算（GPU 与 CPU 看到的是同一份字节）
    std::vector<float> ref(size_t(M) * N);
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t n = 0; n < N; ++n) {
        float acc = 0.f;
        for (uint32_t k = 0; k < K; ++k)
          acc += float(a[size_t(m) * K + k]) * float(b[size_t(n) * K + k]);
        ref[size_t(m) * N + n] = acc;
      }

    std::printf("\n[SWEEP  ] fp8 base_offset 语义扫描（0..7），K=256\n");
    for (uint32_t base = 0; base < 8; ++base) {
      launch(dA.get(), dB.get(), dC.get(), M, N, K, base, K / 32);
      auto rep = hopper::check_close(dC.download(), ref, 3e-2, 3e-2);
      std::printf("  base_offset=%u : %s (max_abs=%.3e)\n", base,
                  rep.pass ? "PASS" : "fail", rep.max_abs_err);
      if (rep.pass && good_base == 0xFFFFFFFF) good_base = base;
    }
    if (good_base == 0xFFFFFFFF) {
      std::printf("  所有 base_offset 均失败！fp8 布局/描述符假设有误\n");
      return 1;
    }

    // ---- 单指令精度探针：K=128 tile 只发射第 1 条 k32 wgmma ----
    // 参考 256 残差 ~1.5e-2。实测单指令 ~7.6e-4（|D|~3 时 ≈2^-12 相对舍入，
    // ~fp22 量级）——精度损失在指令内部累加；K=256 残差超线性（幅值相关
    // 舍入，后期指令部分和大、误差按幅值放大）。详见 docs/notes/07_fp8.md。
    // （不能直接用 K=32 tile：SW128 原子必须满宽 128 元素，tile 越界=ULF）
    {
      constexpr uint32_t K1 = 128;
      auto ha1 = hopper::make_random_vector(size_t(M) * K1, 805, -1.f, 1.f);
      auto hb1 = hopper::make_random_vector(size_t(N) * K1, 806, -1.f, 1.f);
      std::vector<__nv_fp8_e4m3> a1, b1;
      for (float f : ha1) a1.push_back(__nv_fp8_e4m3(f));
      for (float f : hb1) b1.push_back(__nv_fp8_e4m3(f));
      hopper::DeviceBuffer<__nv_fp8_e4m3> dA1(a1.size()), dB1(b1.size());
      hopper::DeviceBuffer<float> dC1(size_t(M) * N);
      dA1.upload(a1);
      dB1.upload(b1);
      std::vector<float> ref1(size_t(M) * N);  // 只算前 32 列（steps=1）
      for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
          float acc = 0.f;
          for (uint32_t k = 0; k < 32; ++k)
            acc += float(a1[size_t(m) * K1 + k]) * float(b1[size_t(n) * K1 + k]);
          ref1[size_t(m) * N + n] = acc;
        }
      launch(dA1.get(), dB1.get(), dC1.get(), M, N, K1, good_base, 1);
      auto rep1 = hopper::check_close(dC1.download(), ref1, 3e-2, 3e-2);
      std::printf("[PROBE  ] 单指令 k32: %s (max_abs=%.3e)  <- 对比 K=256 的 1.5e-2\n",
                  rep1.pass ? "PASS" : "fail", rep1.max_abs_err);
    }
  }

  // ---- 吞吐: 2048x2048x{128,512}，K=128 对照 fp16 SW128 的 35.2T ----
  std::filesystem::create_directories("results");
  hopper::CsvWriter csv("results/wgmma_fp8.csv", {"case", "min_ms", "tflops"});
  for (uint32_t K : {128u, 512u}) {
    constexpr uint32_t MB = 2048, NB = 2048;
    auto ha = hopper::make_random_vector(size_t(MB) * K, 803, -1.f, 1.f);
    auto hb = hopper::make_random_vector(size_t(NB) * K, 804, -1.f, 1.f);
    std::vector<__nv_fp8_e4m3> a, b;
    a.reserve(ha.size());
    b.reserve(hb.size());
    for (float f : ha) a.push_back(__nv_fp8_e4m3(f));
    for (float f : hb) b.push_back(__nv_fp8_e4m3(f));
    hopper::DeviceBuffer<__nv_fp8_e4m3> dA(a.size()), dB(b.size());
    hopper::DeviceBuffer<float> dC(size_t(MB) * NB);
    dA.upload(a);
    dB.upload(b);

    const double flops = 2.0 * MB * NB * K;
    auto st = hopper::time_reps(3, 10, [&] {
      launch(dA.get(), dB.get(), dC.get(), MB, NB, K, good_base, K / 32);
    });
    std::printf("[WGMMA ] 2048x2048x%u fp8 SW128 (base=%u): min %.3f ms -> %.2f TFLOPS\n",
                K, good_base, st.min_ms, hopper::tflops(flops, st.min_ms));
    csv.row({"bench 2048x2048x" + std::to_string(K) + " fp8 sw128",
             std::to_string(st.min_ms),
             std::to_string(hopper::tflops(flops, st.min_ms))});
  }
  return 0;
}
