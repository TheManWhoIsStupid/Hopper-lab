# hopper-lab

NVIDIA Hopper 架构 (sm_90) 特性研究仓库。以**手写 CUDA + 内联 PTX** 的方式逐个吃透
Hopper 的核心硬件特性，最终目标是做出能对标 cuBLAS 的 GEMM，并沉淀可复用的研究笔记。

主线: **基础搬运 → 异步流水线 → wgmma → 综合 GEMM**。

## 环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA H20-3e × 2 (Hopper, cc 9.0, 144GB HBM3e) |
| Driver | 580.126.09 (CUDA 13.0) |
| Toolkit | CUDA 12.9 (V12.9.41, `/usr/local/cuda`) |
| 编译 | GCC 11.4 / CMake 3.22 / `-arch=sm_90a` |

> ⚠️ 本机两块 GPU 常年被任务占满（利用率 100%），benchmark 数据以 `min_ms` 为准，
> 关键实验挑空闲时段重跑。

## 研究路线

- [x] **Phase 0** 框架搭建：device query / timer / checker / bench harness（vectorAdd 冒烟测试）
- [x] **Phase 1** 数据搬运（[01_memcpy.md](docs/notes/01_memcpy.md)）
  - [x] 1a: naive → `cp.async` → TMA 1D bulk 带宽对比
  - [x] 1b: TMA 2D tensor map + SWIZZLE_128B 布局（公式已 bit-exact 验证）
- [x] **Phase 2** `mbarrier`：expect_tx + producer-consumer 多级流水线（[02_pipeline.md](docs/notes/02_pipeline.md)）
- [ ] **Phase 3** `wgmma`：m64n64k16 起步，A/B 操作数 swizzle 布局，逐步对标 cuBLAS
- [ ] **Phase 4** cluster + DSMEM：2-CTA cluster 跨 CTA 共享内存访问
- [ ] **Phase 5** 综合 GEMM：TMA + wgmma + 多级流水线 + warp specialization
- [ ] **Phase 6** 进阶（可选）：FP8 GEMM / persistent kernel / attention

## 构建与运行

```bash
./scripts/run.sh smoke_vector_add        # 构建并运行指定 target
./scripts/run.sh                         # 默认跑冒烟测试
```

或手动:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/src/00_smoke/smoke_vector_add
```

## 目录约定

```
include/common/   # 公共工具: errors / device_info / tensor / timer / checker / bench
src/NN_topic/     # 每个实验一个目录, 产出: 可执行文件 + CHECK + BENCH + CSV
results/          # bench 数据落盘 (gitignore, 防止误提交大量数据)
scripts/          # 构建/运行/profiling 封装
docs/notes/       # 研究笔记: PTX 指令行为、布局图、踩坑记录、实验数据表
third_party/      # CUTLASS 等参考实现 (submodule, 仅作对照学习)
```

**每个实验必须包含三要素**：正确性检查（对照 reference）、性能数据（带宽或 TFLOPS +
峰值百分比）、`docs/notes/` 对应笔记。数据可复现（固定随机种子）。

## 已有笔记

- [00_environment.md](docs/notes/00_environment.md) — 环境记录
- [01_memcpy.md](docs/notes/01_memcpy.md) — 数据搬运三机制对比（naive / cp.async / TMA）
- [02_pipeline.md](docs/notes/02_pipeline.md) — mbarrier 多级流水线（含反面教材）
