// mbarrier + TMA 装载的 PTX 封装（Phase 1b/2 实验中验证过的模式提升为公共头）。
// 语义要点:
//   * full barrier: arrive count=1，producer 的 arrive.expect_tx 同时完成一次 arrive
//     并登记预期 tx 字节数；TMA 完成 时记账，两者齐 -> phase 翻转
//   * empty barrier: arrive count=消费者数，全体消费完各自 arrive -> phase 翻转
//   * 奇偶跟踪: stage 的第 k 次使用对应第 k 个 phase，parity = k & 1（k = i / S）
#pragma once

#include <cuda.h>

#include <cstdint>

#include "common/gmma.cuh"

namespace hopper {

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)),
               "r"(count));
}

// producer: 登记 tx 并完成一次 arrive（full barrier 记账起点）
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar,
                                                          uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
          smem_u32(bar)),
      "r"(bytes));
}

// consumer: 纯 arrive（empty barrier 计数）
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)));
}

// 自旋等待指定奇偶的 phase 完成（{} 作用域让 label 局部化，CUTLASS 同款手法）
__device__ __forceinline__ void mbarrier_wait_parity(const uint64_t* bar,
                                                     uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t}\n" ::"r"(smem_u32(bar)),
      "r"(parity));
}

// TMA 2D tile 装载: 从 tmap 的 (c0,c1)（元素坐标, c0=内维）取一个 box 到 smem，
// 完成时向 bar 记账 tx 字节
__device__ __forceinline__ void tma_load_2d(void* smem_dst,
                                            const CUtensorMap* tmap, int32_t c0,
                                            int32_t c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
          smem_u32(smem_dst)),
      "l"(tmap), "r"(c0), "r"(c1), "r"(smem_u32(bar)));
}

// ---- TMA 写出（bulk async group 语义，与 load 的 mbarrier 记账是两套机制）----
// store 不走 mbarrier：进度由 per-thread 的 bulk async-group 跟踪，
// commit_group 收拢、wait_group.read 确认源 smem 已读完（可复用/可退出）。

// smem -> tmap 的 (c0,c1) 处写一个 box（注意与 load 的操作数顺序相反：目的在前）
__device__ __forceinline__ void tma_store_2d(const void* smem_src,
                                             const CUtensorMap* tmap, int32_t c0,
                                             int32_t c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
      " [%0, {%1, %2}], [%3];\n" ::"l"(tmap),
      "r"(c0), "r"(c1), "r"(smem_u32(smem_src))
      : "memory");
}

__device__ __forceinline__ void tma_store_commit_group() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

// 等到最多还剩 N 组未"读源完毕"（smem 安全可复用）
template <int N>
__device__ __forceinline__ void tma_store_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
}

}  // namespace hopper
