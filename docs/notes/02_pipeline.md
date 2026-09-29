# 02 — mbarrier 多级流水线 (producer-consumer)

**日期**: 2026-09-29 · **代码**: `src/02_pipeline/` · **数据**: `results/pipeline.csv`

## 结构

S 个 16KB stage 组成环形缓冲，每 stage 一对 mbarrier：

```
full[s]   arrive count = 1    producer(tid0) arrive.expect_tx + TMA complete_tx
empty[s]  arrive count = 256  全体消费者消费完 arrive，producer 等它再覆写

producer 领先消费者 S-1 个迭代:  迭代 i 开头 issue(i+S-1)，消费 i 时
                                后面 S-1 个 TMA 在飞  ← 流水线的关键
```

**奇偶算术**: stage 的第 k 次使用对应其 barrier 的第 k 个 phase，parity = k&1，
其中 `k = i / S`（i 为本 block 的迭代序号）。empty 的等待要落后一轮：
parity = `((i/S) - 1) & 1`，且 i < S 时跳过（首轮天然空闲，priming）。

## ⭐ 反面教材（本实验最大的收获）

第一版把 issue 和 wait 放在**同一迭代**里（issue(i) 后立刻 wait full(i)），
结果 S=1 反而最快、S≥2 持平甚至更慢——因为 load 延迟**完全暴露**，stage 数
毫无作用。git 历史里有一版数据（1b964f5 之前的 commit）：

| 结构 | S=1 @4/SM | S=2 @4/SM |
|------|-----------|-----------|
| 同迭代 issue+wait（错误） | 1616 GB/s | 1403 GB/s |
| producer 领先 S-1（正确） | 1443 GB/s | **1809 GB/s** |

结论：**多级流水线的收益全部来自"提前发 load"，缓冲区本身只是让提前成为可能**。
（S=1 在错误结构下"更快"是因为它不浪费 smem 和额外的 barrier 往返。）

## 实验数据（满载 GPU）

tile 16KB，消费 = 每线程 128 次伪随机 smem 读（8× 复用，刻意与搬运同量级）：

| S \ grid | 78 (1/SM) | 156 (2/SM) | 312 (4/SM) |
|----------|-----------|------------|------------|
| 1 | 931 GB/s (19.3%) | 1445 (30.0%) | 1443 (30.0%) |
| 2 | 1386 (28.8%) | **1815 (37.7%)** | 1809 (37.6%) |
| 3 | 1404 (29.2%) | 1811 | 1824 (37.9%) |
| 4 | 1406 | 1804 | 1819 |
| 6 | 1402 | 1806 | 1822 |

## 分析

1. **双重缓冲立竿见影**: 1 blk/SM 时 +49%（931→1386），2-4 blk/SM 时 +25%。
   块内流水线是低占用率下唯一能藏 DRAM 延迟的手段。
2. **S>2 几乎无增益**: 稳态耗时 = max(load, consume)，两者恒定时更深缓冲无事可做。
   真实 GEMM 里 stage 数的价值在于吸收**抖动**（epilogue、k-尾部、occupancy 波动），
   以及更大的 tile 带来的更平滑供给——CUTLASS 选 3-4 级是在 smem 容量
   （227KB 上限）和鲁棒性之间的折中。
3. **smem 容量约束的实证**: S=4 每块 65.6KB，4 blk/SM 需 262KB > 228KB → 实际
   只有 3 块常驻（第四块排队），但性能不降——BW-bound 场景对占用率不敏感。
   这解释了为什么大 tile + 少 stage 是安全的，而 compute-bound 场景不行。
4. 正确性验证顺便证明了**流水线不改变数值语义**（bit-exact，同一累加顺序）。

## 沉淀的模板

`pipeline_kernel<S>` 即 CUTLASS `PipelineTmaAsync` 的最小可读版本，Phase 5 的
综合 GEMM 直接在此骨架上把"消费"换成 wgmma 即可。
