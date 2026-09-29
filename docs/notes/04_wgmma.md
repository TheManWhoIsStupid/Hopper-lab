# Phase 4: wgmma (warpgroup MMA)

> 源码: `src/04_wgmma/wgmma_basic.cu` (4a), `src/04_wgmma/wgmma_swizzle128.cu` (4b)
> 数据: `results/wgmma_basic.csv`, `results/wgmma_sw128.csv`
> 权威参照: CUTLASS `cute/arch/mma_sm90_desc.hpp`, `mma_sm90_gmma.hpp`, `atom/mma_traits_sm90_gmma.hpp`
> （/tmp/cutlass 浅克隆，SSH 通 GitHub）

## 机制速查（已源码级验证）

### wgmma.mma_async m64n64k16 fp16
```
wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
  {d0..d31}, a_desc, b_desc, p, 1, 1, 0, 0;
```
- p = scale-d 谓词（0 覆盖 / 1 累加），随后 scale-a/b、trans-a/b 均立即数
- **trans 标志**: CUTLASS `enum Major { K=0, MN=1 }` → A/B 均 K-major 存储时 **trans-a=trans-b=0**
  （B 按 N×K 转置存放）
- 配套: `wgmma.fence.sync.aligned`（写累加寄存器后）、`wgmma.commit_group`、`wgmma.wait_group 0`、
  `fence.proxy.async.shared::cta`（通用 proxy 写 → wgmma 异步 proxy 可见，**必加**）

### 矩阵描述符 64 位位域
| bits | 字段 | 说明 |
|---|---|---|
| [0,14) | start_addr >> 4 | smem 起始（16B 对齐） |
| [16,30) | LBO >> 4 | leading byte offset |
| [32,46) | SBO >> 4 | stride byte offset |
| [49,52) | base_offset | 仅 SW128/SW64 有效 |
| [62,64) | swizzle | **0=none, 1=128B, 2=64B, 3=32B** |

K-major 规范（uint128 单位，CUTLASS make_gmma_desc 注释）：
- INTERLEAVE（无 swizzle）: `((8,n),2):((1,SBO),LBO)` —— **8×8 核心矩阵连续 128B**（行距固定 16B），
  LBO=沿 K 核心矩阵步距（自由），SBO=沿 M/N 8 行组步距（自由）
- SW128: `((8,n),2):((8,SBO),1)` —— 8×64(fp16) swizzle 原子（行 128B，XOR 公式同 Phase 1b），
  行距 128B，**LBO 字段固定为 1**，SBO=原子步距 1024B

### 累加器映射（CLayout_64xN，m64n64 = 每线程 32 个 f32）
```
warp = tid/32, lane = tid%32, r = 寄存器号 0..31
m = 16*warp + lane/4 + 8*((r/2)%2)
n = 2*(lane%4) + r%2 + 8*(r/4)
```

## 4a: 无 swizzle 基线（wgmma_basic.cu）

每 block 一个 warpgroup 算 64×64 tile，K=128 整段驻留 smem，8 拍 wgmma。
smem 按核心矩阵布局：`(i,j)` 核心矩阵（8 行×8 列）连续 128B，位于 `i*16K + j*128` 字节。

- 正确性: 512×512×128 vs CPU，rtol=atol=1e-3，**PASS**（max_abs 9.5e-6）
- 吞吐: 2048×2048×128 → **27.39 TFLOPS**（无流水线、单 stage、满载 GPU 争抢下）

## 4b: SW128 swizzle（wgmma_swizzle128.cu）

布局: 元素 (m,k) → `(k/64*8 + m/8)*1024 + (m%8)*128 + ((k%64/8 ^ (m%8))*16 + k%64%8*2)` 字节
（与 Phase 1b TMA SWIZZLE_128B 公式同源——**同一公式服务 TMA 写入与 wgmma 读取**，这是
Phase 6 流水线的接口基石）。

### base_offset 经验扫描（K=128，k16 步进相位 0/2/4/6）

| base_offset | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 结果 | **PASS** | fail | fail | fail | **PASS** | fail | fail | fail |

- **base_offset=0 正确**：与 CUTLASS DescriptorIterator 行为一致——k16 步进只需推进
  start_address 字段（+32B/步），硬件按地址位自行恢复 swizzle 相位。
- base=4 意外也 PASS：推测其 XOR 相位置换（{0,2,4,6}→{4,6,0,2}）对 A/B 两个操作数对称，
  k 求和顺序不变；奇数 base 破坏 8 元素对齐，必然错。**用 0。**

### 吞吐

| 布局 | TFLOPS (2048³, K=128, 争抢下) |
|---|---|
| INTERLEAVE 无 swizzle | 27.39 |
| SW128 | **35.25 (+28.6%)** |

## 结论与教训

1. **swizzle 收益实测 +28.6%**——即便在无流水线 kernel 里，wgmma 的 smem 读取也吃 bank conflict 代价；
   Phase 1b 学的公式在 wgmma 侧闭环。
2. wgmma 上手三件套（描述符位域 / K-major 规范 / 累加器映射）全部可以从 CUTLASS 源码
   直接读出，不必猜。
3. `fence.proxy.async.shared::cta` 不可省——普通 st.shared 写的 smem 对 wgmma 不可见。
4. 35.25T 已达争抢态 mma.sync 微基准 (45.2T) 的 78%、争抢态 cuBLAS HGEMM (72.6T) 的 49%——
   **无流水线、K 驻留单 stage 的结构上限就在这里**。突破口是 Phase 6：TMA 多级流水线 +
   wgmma 异步重叠（当前 8 拍 wgmma 串行等完 + 每 tile 重装 smem）。

## 机器级坑（重要）

- **`-arch=sm_90a` 在本机 nvcc 12.9 下静默降级 `compute_90`**：TMA 不受影响（sm_90 就支持），
  但 wgmma f16 报 "not supported on .target 'sm_90'"。必须用
  `-gencode=arch=compute_90a,code=sm_90a`。已在顶层 CMakeLists 修正（此前所有二进制实际
  都是 sm_90 目标——纯侥幸没炸）。
