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

## 待办 / 后续

- [ ] GPU 空闲时复测绝对值
- [ ] **Phase 1b: TMA 2D tensor map + swizzle**（`cuTensorMapEncodeTiled` +
      `cp.async.bulk.tensor.2d`，SWIZZLE_128B 的 smem 地址变换规则——wgmma 的前置知识）
- [ ] ncu 验证: `dram__bytes.sum` 与理论搬运量对账
