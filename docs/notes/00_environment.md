# 00 — 环境记录

**日期**: 2026-09-29

## 硬件 / 软件环境

| 项目 | 值 | 备注 |
|------|-----|------|
| GPU | NVIDIA H20-3e × 2 | Hopper (GH 架构), cc 9.0, 144GB HBM3e |
| Driver | 580.126.09 | 支持 CUDA 13.0 |
| Toolkit | CUDA 12.9 (V12.9.41) | `/usr/local/cuda` → cuda-12.9；另有 12.4/12.6/12.8 可用 |
| CPU 侧 | GCC 11.4, CMake 3.22, Python 3.11 | Ubuntu 22.04 |

## H20-3e 对研究的影响

- **sm_90a 可用**: TMA (`cp.async.bulk.tensor`), wgmma, thread block cluster,
  DSMEM, FP8 (e4m3/e5m2) 全部可用，编译用 `-arch=sm_90a`。
  （注意: sm_90a 编译的二进制只能在 Hopper 上跑，不带 `a` 的 sm_90 缺这些指令。）
- **H20 的算力规格与 H100 不同**（砍了 tensor core 峰值），但显存带宽是强项。
  结论: memory-bound 实验在这台机器上很有代表性；compute-bound 的 wgmma 实验
  绝对数值不能和 H100 论文数据直接比，要看相对趋势。
- **两卡常被占满** (util 100%, 显存 ~140/143GB): benchmark 噪声大，正式数据
  需挑空闲时段；`time_reps` 取 min_ms 也是为了抗噪。

## 实测设备属性（Phase 0 冒烟测试，2026-09-29）

| 属性 | 值 |
|------|-----|
| SM 数量 | 78 @ 1980 MHz |
| 显存 | 139.8 GB（标称 144GB），L2 60 MB |
| shared mem / block | 48 KB 默认 / **227 KB opt-in**（Phase 1+ 大 tile 需要） |
| 显存总线 | 6016-bit @ 3.20 GHz，公式估算 ~4814 GB/s（实测参考值，见下行） |
| cluster launch | 支持 ✅ |

vector_add 冒烟（GPU 满载状态下）：2376 GB/s ≈ 峰值的 49%——
干净环境下的 memory-bound 基线数据待空闲时补测。

## 工具链备忘

- **坑**: PATH 里的 `/usr/bin/nvcc` 是 apt 装的 CUDA 11.5（不支持 sm_90a），CMake
  会优先抓到它。因此 `scripts/run.sh` 里显式 export `CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`。
- CMake 3.22 < 3.24，不认识 `90a` 架构值 → 顶层 CMakeLists 里
  `CMAKE_CUDA_ARCHITECTURES=OFF` + 手动 `-arch=sm_90a`。
- ncu / nsys 的可用性待确认（Phase 1 用到前先 `which ncu nsys` 检查）。
- 机器上还有 nccl / nvshmem / LeetCUDA 等目录，可作参考但不依赖。
