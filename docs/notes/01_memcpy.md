# 01 — 数据搬运: naive vs cp.async vs TMA

**日期**: 2026-09-29 · **代码**: `src/01_memcpy/` · **数据**: `results/copy_compare.csv`

## 实验设计

三种 global→shared 搬运机制，控制变量后对比:

| 变量 | 值 |
|------|-----|
| tile 大小 | 32KB（单缓冲，逐 tile 流水） |
| block 规模 | 256 线程 |
| 总数据量 | 256MB（8192 个 tile，grid-stride 分配） |
| 消费方式 | 每 tile 由 tid0 从 smem 旋转位置读 1 个 float 写回 global（防 DCE + 三机制成本一致） |
| 扫描维度 | grid = 78×{1,2,4,7}（H20-3e 共 78 SM；7×32KB=224KB≈smem 上限） |

## 三种机制用到的 PTX

```
// naive: 传统 LDG -> 寄存器 -> STS，编译器自动生成

// cp.async (Ampere): 异步 global -> shared，不占寄存器
cp.async.cg.shared.global [smem], [gmem], 16;   // 16B/线程/条, .cg 只过 L2
cp.async.commit_group;
cp.async.wait_group 0;                          // 等本线程所有组到齐

// TMA bulk 1D (Hopper): tid0 一条指令搬整个 32KB tile
mbarrier.init.shared::cta.b64 [bar], 1;             // arrive count = 1
mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], 32768;  // 预期收 32KB
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem], [gmem], 32768, [bar];                   // 完成后向 bar 记账
// 消费者: mbarrier.try_wait.parity 自旋等 phase 完成（phase 奇偶交替 0/1/0/1...）
```

**mbarrier 语义速记**: 一个 phase 完成 ⇔ (arrive 次数到齐 **且** expect_tx 的字节数
被 TMA complete_tx 记满)。phase 每完成一次奇偶翻转一次，`try_wait.parity` 的参数
是"要等的那个 phase 的奇偶"。单缓冲下只有 leader 读写 smem，靠程序序保证 WAR 安全。

## 实验数据（GPU 满载状态下，看相对关系）

| mech | 78 blocks (1/SM) | 156 (2/SM) | 312 (4/SM) | 546 (7/SM) |
|------|------------------|------------|------------|------------|
| naive | 2189 GB/s (45.5%) | 2877 (59.8%) | 3234 (67.2%) | 3022 (62.8%) |
| cp.async | 2680 (55.7%) | **3810 (79.1%)** | **3857 (80.1%)** | 3455 (71.8%) |
| tma1d | 2612 (54.2%) | 3787 (78.7%) | 3806 (79.1%) | 3431 (71.3%) |

（峰值按公式估算 ~4814 GB/s；两卡被其他任务占满，绝对值偏低）

## 结论

1. **异步机制（cp.async / TMA）用 2 blocks/SM（512 线程/SM）就打到 ~80% 峰值**，
   naive 要 4 blocks/SM 且止步 ~67%。绕过寄存器 + 不阻塞的搬运在"每线程能供养的
    outstanding 字节数"上有数量级优势。
2. **单级流水下 cp.async ≈ TMA（差距 <2%）**——TMA 的卖点不是单拷贝带宽，而是:
   - 指令效率: 每 32KB tile，cp.async 要 2048 条指令(256线程×8)，TMA 只要 1 条
   - 不占线程: TMA 只需 tid0 发起，解放其余 255 线程（Phase 5 warp specialization 的基础）
   - 这两者在高频小 tile + 多阶段的 GEMM 场景才真正拉开差距 → Phase 2 流水线验证
3. **7 blocks/SM 反而变慢**（3.8→3.4TB/s）：可能是满载 GPU 的噪声，也可能是
   224KB smem 吃满导致 L1/smem 资源争用。待空闲时复测确认。
4. naive 在 1 block/SM 时最惨（45%）：256 线程×16B outstanding = 4KB/SM 在飞，
   远不够填饱 HBM3e。

## Phase 1b: TMA 2D tensor map + swizzle

**代码**: `src/01_memcpy/tma2d_swizzle.cu` · **数据**: `results/tma2d_swizzle.csv`

### 实验设计

8192×8192 fp32 矩阵，box = **32 列 × 128 行**（内维 32×4B = **128B，正好等于
swizzle 宽度**），每 box 16KB。对比 `SWIZZLE_NONE` vs `SWIZZLE_128B`。

消费方式特意选了 **列读**（每 warp 32 线程读同列不同行）——NONE 下 32 线程全部
落在同一 bank（32-way conflict），128B swizzle 把它们散到 8 个 chunk（4-way）。

### 关键 API / 指令记录

```cpp
// host: 描述"怎么搬"的 descriptor
cuTensorMapEncodeTiled(&tmap, FLOAT32, /*rank*/2, gmem_ptr,
                       /*globalDim*/{8192cols, 8192rows},   // dim[0] 是最内维!
                       /*globalStrides*/{32768B},           // 只给 dim1.., 必须 16 的倍数
                       /*boxDim*/{32, 128},
                       /*elementStrides*/{1,1}, INTERLEAVE_NONE,
                       swizzle /* NONE 或 128B */, L2_128B, OOB_FILL_NONE);

// kernel 参数必须 __grid_constant__，PTX 里用 [&tmap] 作 descriptor 操作数
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [smem], [tmap, {x, y}], [bar];    // 坐标是元素单位
```

### SWIZZLE_128B 的 smem 地址公式（本文档最重要的一条 ⭐）

TMA 写 smem 时，行内 16B chunk 序号会与行号做异或（CUTLASS 的 `Swizzle<3,4,3>`）:

```
smem_word(row, col) = row * (128B/4B) + ( ((col/4) ^ (row%8)) * 4 + col%4 )
                                          ~~~~~~~~~~~~~~~ 16B chunk 变换
```

**验证方式**: kernel 用该公式从 swizzled smem 读值写回，与 host 期望 bit-exact 比对
→ PASS ✅。这个公式就是将来 wgmma 从 smem 读操作数时的布局（也是调试 CUTLASS
共享内存布局的钥匙）。约束: box 内维字节 = swizzle 宽度的倍数，smem 基址 128B 对齐。

### 数据（满载 GPU）

| mode | 78 (1/SM) | 156 (2/SM) | 312 (4/SM) | 624 (8/SM) |
|------|-----------|------------|------------|------------|
| swizzle_none | 1334 GB/s (27.7%) | 2279 (47.3%) | 3349 (69.6%) | 3684 (76.5%) |
| swizzle_128B | 1492 (31.0%) | 2446 (50.8%) | 3442 (71.5%) | 3657 (76.0%) |
| **加速比** | **+11.8%** | +7.3% | +2.8% | ≈0 |

### 分析

1. **bank conflict 的代价在低占用率时最明显**（+12%）——此时没有足够的并行
   warp 来隐藏冲突延迟；占用率升高后延迟被隐藏，差距消失。结论: swizzle 不是
   可选项，真实 GEMM 里 smem 读取远比本实验密集，没有 swizzle 的 K-major 布局
   是不可用的。
2. 本实验消费太轻（每 16KB box 只读 1KB），冲突代价被稀释——纯 smem bank
   conflict 微基准留作独立实验（Phase 3 wgmma 前做更有意义）。
3. **16KB box (3.66TB/s) vs 1a 的 32KB tile (3.86TB/s)**: box 越小，单位数据的
   TMA 发射 + mbarrier 等待次数越多。GEMM tile 设计要在"流水线粒度"和"smem
   容量"之间权衡（Phase 5 会回到这点）。

## 待办 / 后续

- [ ] GPU 空闲时复测绝对值
- [ ] ncu 验证: `dram__bytes.sum` 与理论搬运量对账; 观察 smem bank conflict 计数器
- [ ] 32B/64B swizzle 模式的公式验证（目前只验证了 128B，够 wgmma 用）
