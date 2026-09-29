// Thread Block Cluster + DSMEM 原语（对齐 cute/arch/cluster_sm90.hpp）。
// 语义要点:
//   * cluster = 一组保证同驻同调度的 CTA（同一 GPC 内不同 SM），可互访 shared memory
//   * mapa: 把本 CTA 的 shared 地址映射为 target rank CTA 的对应地址（DSMEM 寻址）
//   * barrier.cluster.arrive/wait 默认带 release/acquire（cluster 作用域），
//     arrive 前的写对 wait 后的所有 CTA 可见
//   * mbarrier.init 之后跨 CTA 使用前须 fence.mbarrier_init.release.cluster
//   * TMA multicast: ctaMask 指定接收 CTA 集合，一条指令数据分发到各 CTA 的
//     同偏移 smem；mbarrier 记账语义见 docs/notes/06_cluster.md 的发射方实验
#pragma once

#include <cuda.h>

#include <cstdint>

#include "common/gmma.cuh"

namespace hopper {

__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
  return rank;
}

// 本 CTA shared 地址 -> target rank CTA 的 DSMEM 地址
__device__ __forceinline__ uint32_t mapa(uint32_t smem_addr, uint32_t target_rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
               : "=r"(r)
               : "r"(smem_addr), "r"(target_rank));
  return r;
}

__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

// mbarrier.init 结果对 cluster 内其他 CTA 可见（远程 arrive 前必备）
__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

// ---- DSMEM 读写（地址为 mapa 结果或本 CTA shared 地址）----

__device__ __forceinline__ uint32_t ld_shared_cluster_u32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared::cluster.u32 %0, [%1];\n" : "=r"(v) : "r"(addr));
  return v;
}

__device__ __forceinline__ void st_shared_cluster_u32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared::cluster.u32 [%0], %1;\n" ::"r"(addr), "r"(v));
}

__device__ __forceinline__ uint4 ld_shared_cluster_u128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared::cluster.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr));
  return v;
}

__device__ __forceinline__ uint4 ld_shared_u128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr));
  return v;
}

// 远程 mbarrier arrive：对 target CTA（mapa 后的地址）的 barrier 记一次 arrive
__device__ __forceinline__ void mbarrier_arrive_remote(uint32_t remote_bar_addr) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" ::"r"(remote_bar_addr)
               : "memory");
}

// TMA 2D multicast 装载：ctaMask 的每个 CTA 在同偏移 smem 各收一份数据。
// 指令串对齐 cute/arch/copy_sm90_tma.hpp SM90_TMA_LOAD_MULTICAST_2D（去 cache_hint）。
__device__ __forceinline__ void tma_load_2d_mcast(void* smem_dst,
                                                  const CUtensorMap* tmap,
                                                  int32_t c0, int32_t c1,
                                                  uint64_t* bar, uint16_t cta_mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global"
      ".mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(smem_u32(smem_dst)),
      "l"(tmap), "r"(c0), "r"(c1), "r"(smem_u32(bar)), "h"(cta_mask));
}

}  // namespace hopper
