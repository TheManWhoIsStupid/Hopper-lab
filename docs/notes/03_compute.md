# Phase 3: 算力密度基线 (compute peak)

> 源码: `src/03_compute/peak_flops.cu`, `src/03_compute/cublas_gemm.cu`
> 数据: `results/peak_flops.csv`, `results/cublas_gemm.csv`

## 为什么测

H20 是"砍算力保带宽"的 Hopper 变体，张量核规格与 H100 差距极大，公开资料口径不一。
Phase 4 (wgmma) / Phase 6 (综合 GEMM) 的所有 TFLOPS 数据都需要一个**实测**的分母，
否则 `%peak` 指标不可信。本阶段一次测齐三层：

| 层 | 测法 | 说明 |
|---|---|---|
| FP32 CUDA core | 8 条独立 FFMA 累加链，纯寄存器 | 可由架构推算对账：78 SM × 128 lane × 2 flop × 1.98 GHz ≈ **39.5 TFLOPS** |
| FP16 tensor | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 寄存器循环 | 每条 4096 flop；Hopper 张量核对 mma.sync 与 wgmma 吞吐一致，仅发射粒度不同 |
| cuBLAS 实际 | HGEMM 8192³ (compute-32F) + SGEMM 4096³ | Phase 6 对标线；先做 64³ fp16 正确性 sanity |

## 方法

- 无内存流量（操作数全在寄存器），测的是发射吞吐上限；
- `grid = 78 × 8` blocks × 256 threads = 2048 线程/SM 满占用；
- mma 依赖链：每 warp 串行依赖 accumulator，靠 64 warps/SM 互掩延迟（标准做法）；
- 时间取 `min_ms`（共享 GPU 满载，绝对值偏低，空闲时复测）。

## 实测数据

### 满载数据点（2026-09-29，sglang 推理服务 util=100% 争抢下，min_ms）

| 项目 | TFLOPS | 参照 | 备注 |
|---|---|---|---|
| FP32 FFMA | 15.89 | 39.5 (架构推算) 的 40% | 争抢严重，仅作下界 |
| FP16 tensor (mma.sync) | 45.21 | 为 FP32 FFMA 的 2.8 倍 | 同上 |
| cuBLAS HGEMM 8192³ | 72.59 | — | 超过我们争抢下的 mma.sync 微基准，说明微基准受争抢影响更大 |
| cuBLAS SGEMM 4096³ | 13.42 | — | |

<!-- 真空闲复测数据由 idle-watch 任务自动补充（条件: 空闲内存≥2.5GB 且 util≤5%） -->

## 结论（初步，待真空复测修正）

- 满载数字只建立**下界**：FP32 ≥ 15.9T、fp16 tensor ≥ 45.2T、HGEMM ≥ 72.6T；
- H20 公开标称 fp16 张量峰值 ~148T（待真空实测验证）；
- 有意思的现象：争抢下 cuBLAS HGEMM (72.6T) 反而高于 mma.sync 寄存器微基準 (45.2T)——
  微基準 grid 满占用 (2048 线程/SM)，与推理服务的 kernel 抢占交替代价更大；
  cuBLAS kernel 占用资源更少/更短，交错损耗更小。教训：**满载机器上的微基準相对排序不可信**。

## 环境记录

- 2026-09-29 13:2x: 8×H20 全被 root 的 `sglang::scheduler_TP0-7`（8 卡 TP 推理服务）占满，
  每卡仅剩 ~700-800MB，连 CUDA context（~1GB）都放不下，任何 GPU 程序无法运行。
- 2026-09-29 14:0x: sglang 释放部分显存（每卡 ~5.9GB 空闲，util 仍 100%），小型实验可跑，
  上述满载数据点在此条件下测得；真空复测由监视任务待机执行（条件加了 util≤5%）。
