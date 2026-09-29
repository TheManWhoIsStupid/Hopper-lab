// Hopper wgmma (warpgroup MMA) PTX 封装与 GMMA 矩阵描述符。
// 位域/规范与 CUTLASS cute/arch/mma_sm90_desc.hpp 及 PTX ISA "Matrix Descriptor Format" 对齐，
// 关键结论（已在 docs/notes/04_wgmma.md 记录推导）：
//   * A/B 均 K-major 存储时 trans-a=trans-b=0（CUTLASS: enum Major { K=0, MN=1 }）
//   * K-major 无 swizzle 规范: ((8,n),2):((1,SBO),LBO)（uint128 单位）
//     —— 8x8 核心矩阵连续存放 128B（行距固定 16B），LBO=沿K步距，SBO=沿M/N 8行组步距
//   * 累加器 (warp, lane, reg r) -> (m, n):
//       m = 16*warp + lane/4 + 8*((r/2)%2);  n = 2*(lane%4) + r%2 + 8*(r/4)
#pragma once

#include <cstdint>

namespace hopper {

// shared memory 泛型地址 -> shared 空间 u32 偏移
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------------------
// GMMA 矩阵描述符
// ---------------------------------------------------------------------------
// swizzle 编码（bits [62,64)）：CUTLASS LayoutType
enum class GmmaSwizzle : uint64_t { None = 0, B128 = 1, B64 = 2, B32 = 3 };

// K-major INTERLEAVE（无 swizzle）描述符。
// smem: 操作数起始地址（16B 对齐）；lbo_bytes/sbo_bytes 须为 16 的倍数。
//   A(MxK): 核心 (i,j)=(8行,8列) 组，字节偏移 i*SBO + j*LBO，核心矩阵内部行距 16B
//   B(NxK): 同构（B 以 N x K 转置存放）
__device__ __forceinline__ uint64_t gmma_desc_k_inter(const void* smem,
                                                      uint32_t lbo_bytes,
                                                      uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= uint64_t((smem_u32(smem) >> 4) & 0x3FFF);          // start addr
  d |= uint64_t((lbo_bytes >> 4) & 0x3FFF) << 16;         // LBO
  d |= uint64_t((sbo_bytes >> 4) & 0x3FFF) << 32;         // SBO
  d |= uint64_t(static_cast<uint64_t>(GmmaSwizzle::None)) << 62;
  return d;
}

// K-major SW128 描述符（Phase 4b 用）。
// 规范(uint128 单位): ((8,n),2):((8,SBO),1) —— 8x128B swizzle 原子, 行距 128B。
//   LBO 字段在 swizzle 布局下固定为 1（CUTLASS 位域注释: "assumed to be 1"）
//   SBO = 8 行组(M/N 方向原子)字节步距
// base_offset: 起点不在 swizzle 原子边界时的相位补偿（仅 SW128/SW64 有效，[0,8)）
__device__ __forceinline__ uint64_t gmma_desc_k_sw128(const void* smem,
                                                      uint32_t sbo_bytes,
                                                      uint32_t base_offset = 0) {
  uint64_t d = 0;
  d |= uint64_t((smem_u32(smem) >> 4) & 0x3FFF);
  d |= uint64_t(1) << 16;                                  // LBO = 1 (swizzle 布局固定)
  d |= uint64_t((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= uint64_t(base_offset & 7) << 49;
  d |= uint64_t(static_cast<uint64_t>(GmmaSwizzle::B128)) << 62;
  return d;
}

// ---------------------------------------------------------------------------
// wgmma 指令封装
// ---------------------------------------------------------------------------

// wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16  D += A*B, A/B 均 smem 描述符
// （asm 字符串与 CUTLASS MMA_64x64x16_F32F16F16_SS 逐字对齐）
// d: 32 个 f32 累加寄存器; scale_d=0 覆盖 D（首个 k 拍），=1 累加
__device__ __forceinline__ void wgmma_m64n64k16_f32_f16(uint64_t desc_a,
                                                        uint64_t desc_b, float* d,
                                                        uint32_t scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31},"
      " %32,"
      " %33,"
      " p,   1, 1, 0, 0;\n"  // scale-a, scale-b, trans-a=K, trans-b=K
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

// wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3  D += A*B, A/B 均 smem 描述符
// （asm 尾串与 CUTLASS MMA_64x64x32_F32E4M3E4M3_SS_TN 逐字对齐）。
// 与 f16 k16 变体的差异：
//   * 尾部操作数只有 scale-a/scale-b（±1 立即数），**无 trans**——8-bit 布局只支持 K-major
//   * 描述符复用 gmma_desc_k_sw128 不变：SW128 原子按字节定义（CUTLASS
//     Layout_K_SW128_Atom_Bits = Swizzle<3,4,3> ∘ (8 行 × 1024bit)），与元素位宽无关；
//     fp8 的 k32 核心矩阵 = 8 行 × 32B，与 fp16 的 k16 核心矩阵字节同构，
//     原子内 k 步进同为 32B（每原子 4 步）
//   * 累加器 (warp, lane, r) -> (m, n) 映射与 f16 m64n64 完全相同
__device__ __forceinline__ void wgmma_m64n64k32_f32_e4m3(uint64_t desc_a,
                                                        uint64_t desc_b, float* d,
                                                        uint32_t scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31},"
      " %32,"
      " %33,"
      " p,   1, 1;\n"  // scale-a, scale-b（fp8 无 trans 操作数）
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

// wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3  D += A*B（8b 用）
// （asm 与 CUTLASS MMA_64x128x32_F32E4M3E4M3_SS_TN 逐字对齐）。
// 与 m64n64k32 唯一差异：64 个累加寄存器——(warp,lane,r)->(m,n) 映射同一
// 公式延拓 r∈[0,64)（n = 2*(lane%4)+r%2+8*(r/4)，r/4 ∈[0,16) ⇒ n∈[0,128)）。
// 单条 FLOP 翻倍（524K vs 262K），B tile 随之 128 宽（16KB/stage）。
__device__ __forceinline__ void wgmma_m64n128k32_f32_e4m3(uint64_t desc_a,
                                                          uint64_t desc_b,
                                                          float* d,
                                                          uint32_t scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31, "
      " %32, %33, %34, %35, %36, %37, %38, %39, "
      " %40, %41, %42, %43, %44, %45, %46, %47, "
      " %48, %49, %50, %51, %52, %53, %54, %55, "
      " %56, %57, %58, %59, %60, %61, %62, %63},"
      " %64,"
      " %65,"
      " p,   1, 1;\n"  // scale-a, scale-b（fp8 无 trans 操作数）
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
        "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
        "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
        "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

// 寄存器屏障：阻止编译器跨 wgmma 异步窗口重排累加寄存器的读写
// （等价 CUTLASS warpgroup_fence_operand）
__device__ __forceinline__ void wgmma_fence_operand(float& reg) {
  asm volatile("" : "+f"(reg)::"memory");
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

// 等待直到最多还有 N 个未完成的 wgmma group（N=1: 保留 1 组在飞，重叠下一轮）
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void wgmma_wait_group_0() { wgmma_wait_group<0>(); }

// 通用 proxy 写 -> 异步 proxy（wgmma/ TMA）可见性围栏
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

}  // namespace hopper
