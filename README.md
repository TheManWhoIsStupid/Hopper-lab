# hopper-lab

NVIDIA Hopper 架构 (sm_90) 特性研究仓库。以**手写 CUDA + 内联 PTX** 的方式逐个吃透
Hopper 的核心硬件特性，最终目标是做出能对标 cuBLAS 的 GEMM，并沉淀可复用的研究笔记。

主线: **基础搬运 → 异步流水线 → wgmma → 综合 GEMM**。

## 环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA H20-3e × 8 (Hopper, cc 9.0, 143.7GB HBM3e) |
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
- [ ] **Phase 3** 算力密度基线（[03_compute.md](docs/notes/03_compute.md)）：FFMA / `mma.sync` 张量核峰值 / cuBLAS 对标
- [ ] **Phase 4** `wgmma`（[04_wgmma.md](docs/notes/04_wgmma.md)）：✅ m64n64k16 正确性 + 描述符位域 + SW128 布局（+28.6%）；待：RS 变体 / 更大 K 深度
- [ ] **Phase 5** 综合 GEMM（[05_gemm.md](docs/notes/05_gemm.md)）：✅ TMA 流水线 + wgmma 重叠（8192³ 追平 cuBLAS 72.2T）；✅ warp specialization + BM=128（C7520 消除，2048³ 117.6T）；✅ TMA store epilogue（L2 写流量 -48%，含 wgmma WAR 竞态复盘）；待：warp-spec × multicast
- [ ] **Phase 6** cluster + DSMEM（[06_cluster.md](docs/notes/06_cluster.md)）：✅ 2-CTA cluster / mapa / 远程 mbarrier / TMA multicast / multicast GEMM；✅ ncu 验证 fetch 减半（L2 读 -17.3% ≈ A 减半模型）；待：2×2 cluster / 与 warp-spec 组合
- [ ] **Phase 7** 进阶（可选）：FP8 GEMM / persistent kernel / attention

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
- [03_compute.md](docs/notes/03_compute.md) — 算力密度基线（FFMA / mma.sync / cuBLAS）
- [04_wgmma.md](docs/notes/04_wgmma.md) — wgmma 指令 / GMMA 描述符位域 / SW128
- [05_gemm.md](docs/notes/05_gemm.md) — 综合 GEMM：TMA 流水线 + wgmma 重叠（追平 cuBLAS）
- [06_cluster.md](docs/notes/06_cluster.md) — cluster/DSMEM/multicast（含记账语义实验）
