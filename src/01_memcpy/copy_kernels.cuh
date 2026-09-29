// Phase 1: global -> shared 三种搬运机制的 kernel。
// 统一变量: tile 32KB / block 256 线程 / grid-stride 逐 tile 处理。
// 每 tile 搬运完成后由 tid 0 读一个旋转位置写入 out（防止 smem 往返被编译器消掉，
// 也让三种机制的消费成本一致）。
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "common/errors.h"

namespace hopper {

constexpr uint32_t kCopyTileBytes = 32768;                       // 32KB / tile
constexpr uint32_t kCopyTileElems = kCopyTileBytes / 4;          // 8192 floats
constexpr uint32_t kCopyBlock = 256;                             // 线程数 / block
constexpr uint32_t kCopyVecsPerTile = kCopyTileBytes / 16;       // 2048 个 float4
constexpr uint32_t kCopyVecsPerThread = kCopyVecsPerTile / kCopyBlock;  // 8

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- 机制 1: naive —— 同步 load 到寄存器再 store 到 shared ----
__global__ void __launch_bounds__(kCopyBlock)
copy_naive_kernel(const float4* __restrict__ in, float* __restrict__ out,
                  uint32_t num_tiles) {
  __shared__ alignas(128) float4 tile[kCopyVecsPerTile];
  const uint32_t tid = threadIdx.x;

  for (uint32_t t = blockIdx.x; t < num_tiles; t += gridDim.x) {
    const float4* src = in + size_t(t) * kCopyVecsPerTile;
#pragma unroll
    for (uint32_t k = 0; k < kCopyVecsPerThread; ++k) {
      // 相邻线程地址连续 -> 128B/8线程 完美合并
      tile[k * kCopyBlock + tid] = src[k * kCopyBlock + tid];
    }
    __syncthreads();
    if (tid == 0) {
      out[t] = reinterpret_cast<const float*>(tile)[(t * 13) & (kCopyTileElems - 1)];
    }
    __syncthreads();  // WAR 保护：下一轮覆写 tile 前确保读完成
  }
}

// ---- 机制 2: cp.async (Ampere) —— 异步 global->shared，不经寄存器 ----
__global__ void __launch_bounds__(kCopyBlock)
copy_cpasync_kernel(const float4* __restrict__ in, float* __restrict__ out,
                    uint32_t num_tiles) {
  __shared__ alignas(128) float4 tile[kCopyVecsPerTile];
  const uint32_t tid = threadIdx.x;

  for (uint32_t t = blockIdx.x; t < num_tiles; t += gridDim.x) {
    const float4* src = in + size_t(t) * kCopyVecsPerTile;
#pragma unroll
    for (uint32_t k = 0; k < kCopyVecsPerThread; ++k) {
      // .cg: 只过 L2 不过 L1（16B 专用变体），流式搬运更合适
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                       smem_addr(&tile[k * kCopyBlock + tid])),
                   "l"(&src[k * kCopyBlock + tid]));
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");  // 等 tile 到齐
    __syncthreads();                           // 跨线程可见性
    if (tid == 0) {
      out[t] = reinterpret_cast<const float*>(tile)[(t * 13) & (kCopyTileElems - 1)];
    }
    __syncthreads();
  }
}

// ---- 机制 3: TMA 1D bulk copy (Hopper) —— tid 0 一条指令搬整个 tile ----
// 引入两个 Hopper 核心指令:
//   mbarrier.arrive.expect_tx  : 登记"本 phase 预期收到 32KB"
//   cp.async.bulk...complete_tx::bytes : 异步搬运，完成后向 mbarrier 记账
// 所有线程 spin 在 mbarrier.try_wait.parity 上等数据到齐（phase 奇偶交替）。
__global__ void __launch_bounds__(kCopyBlock)
copy_tma1d_kernel(const float* __restrict__ in, float* __restrict__ out,
                  uint32_t num_tiles) {
  __shared__ alignas(128) float tile[kCopyTileElems];
  __shared__ alignas(8) uint64_t bar;
  const uint32_t tid = threadIdx.x;

  if (tid == 0) {
    // arrive count = 1：每 phase 只有 leader arrive 一次（arrive.expect_tx 合体）
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_addr(&bar)));
  }
  __syncthreads();

  uint32_t i = 0;  // 本 block 处理的第 i 个 tile，对应 mbarrier phase 奇偶 = i & 1
  for (uint32_t t = blockIdx.x; t < num_tiles; t += gridDim.x, ++i) {
    if (tid == 0) {
      asm volatile(
          "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
              smem_addr(&bar)),
          "r"(kCopyTileBytes));
      const float* src = in + size_t(t) * kCopyTileElems;
      asm volatile(
          "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
          " [%0], [%1], %2, [%3];\n" ::"r"(smem_addr(tile)),
          "l"(src), "r"(kCopyTileBytes), "r"(smem_addr(&bar)));
    }
    // { } 块作用域让 PTX label 局部化，多实例不冲突（CUTLASS 同款写法）
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n" ::"r"(smem_addr(&bar)),
        "r"(i & 1));
    if (tid == 0) {
      out[t] = tile[(t * 13) & (kCopyTileElems - 1)];
    }
    // 单缓冲 + 只有 tid0 读 smem：tid0 的程序序保证 WAR 安全，无需额外同步
  }
}

}  // namespace hopper
