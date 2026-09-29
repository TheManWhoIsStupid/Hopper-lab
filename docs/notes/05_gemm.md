# Phase 5: 综合 GEMM（TMA 流水线 + wgmma 异步重叠）

把前四个阶段的部件拼成完整 GEMM：**Phase 1b 的 TMA SW128 装载 + Phase 2 的 mbarrier
多级流水线 + Phase 4b 的 wgmma SW128 描述符**。这是本仓库主线目标的第一个里程碑——
从零手写的内核追平 cuBLAS（contended 同条件对比）。

代码: `src/05_gemm/gemm_fused.cu`，公共部件提升到 `include/common/mbarrier.cuh`。

## 内核结构

```
C[128×128 grid] = A(M×K) × B(N×K)ᵀ   全部 row-major fp16 入，fp32 出
tile: BM=BN=BK=64, block=128 线程 = 1 warpgroup
smem: S 级 stage，每级 A 8KB + B 8KB（SW128 规范布局，1024B 对齐）
      + 2S 个 mbarrier（full[S]: count=1 记账型, empty[S]: count=128）
```

- **producer**: tid0 独任。`issue(j)` = wait empty[j%S]（首轮跳过）→
  `arrive_expect_tx(16384)` → 两条 `tma_load_2d`（A/B 各一 box，同一 full barrier 记账）
- **consumer**: 全体 128 线程。wgmma ×4 拍（k16 步进 = start_addr +32B，base_offset=0）
- **TMA 与 wgmma 走同一个 SW128 公式**：Phase 1b 验证过 TMA 写入布局，
  Phase 4b 验证过 wgmma 读取布局，本次端到端闭环——正确性 PASS 证明两侧相位一致
  （swizzle 相位由地址位硬件恢复，两侧共用同一绝对地址即自洽）。

## 流水线时序（关键设计决策）

```
prologue: tid0 发 j = 0..S-2            （S-1 个 load 在飞）
迭代 i:   wait full[i%S]                 （数据到位）
          4× wgmma k16 → commit_group
          if i≥1: wait_group<1>          （上一组读完它的 stage）
                  arrive empty[(i-1)%S]   （全体 128 线程）
          tid0: issue(i+S-1)             （empty 刚翻转，立即复用该 stage）
epilogue: wait_group<0> → 累加器写出
```

两个反直觉的点，都是死锁/竞争分析的结果：

1. **issue 放迭代末尾**（Phase 2 放开头）。tid0 既是 producer 又是 consumer：若在
   迭代开头 issue，它要 wait 的 empty[(i-1)%S] 恰是自己本轮稍后才 arrive 的——自锁。
2. **wait_group<1> 滞后一拍释放**。wgmma 对 smem 的读取是异步的，commit ≠ 读完。
   保留 1 组在飞（跨迭代边界重叠），等到下一组已 commit 后再确认上一组读完，才 arrive
   empty。S=1 在此结构下结构性死锁（prologue 无可发、empty 永不翻转），已从扫描中剔除。

## 实验数据（contended，min_ms）

正确性: 512×512×256, S∈{2,3,4} 全部 PASS（mismatch=0, max_abs=2.3e-5）。

2048×2048×1024（8.59 GFLOP/launch）:

| stages | min_ms | TFLOPS |
|-------:|-------:|-------:|
| 2      | 0.083  | **103.6** |
| 3      | 0.089  | 96.6  |
| 4      | 0.089  | 97.0  |
| 6      | 0.093  | 92.4  |
| 8      | 0.146  | 58.9  |

8192×8192×8192（cuBLAS HGEMM contended 基线 72.59T，Phase 3）:

| stages | min_ms | TFLOPS |
|-------:|-------:|-------:|
| 2      | 15.24  | **72.1** |
| 3      | 15.22  | **72.2** |
| 4      | 15.44  | 71.2  |
| 6      | 18.24  | 60.3  |
| 8      | 33.02  | 33.3  |

## 分析

- **追平 cuBLAS**: 8192³ 手写 72.2T vs cuBLAS 72.59T（同 GPU 争用条件的两个下界）。
  注意 cuBLAS 面向干净 GPU 优化，此处只说明同环境下打平；真空复测待空闲窗口。
- **比无流水线快 2.9×**: Phase 4b（gmem→smem 同步拷贝 + wgmma）35.25T → 103.6T。
  纯粹来自 TMA 异步预取 + wgmma 异步重叠，计算指令一字未改。
- **S=2 反直觉地最优**：lookahead 只有 1 个 stage 却够用——每 stage 16KB，
  wgmma 4 拍耗时足够 TMA 完成 16KB 装载；更大的 S 反而占 smem、降 occupancy、
  增 L2 footprint。
- **S=8 崩塌（33T）是 occupancy**：smem = 2×8×8KB+barriers ≈ 131KB → 1 CTA/SM →
  每 SM 仅 4 warp（上限 64）。S=2 时 32.8KB → 6 CTA/SM → 24 warp，延迟藏得住。
  **stage 数不是免费的：smem 预算直接换 CTA 驻留数**。

## 已知问题 / 记录

- ptxas **C7520**: "wgmma serialized due to compiler-inserted WG.AR in divergent
  path"——单 warpgroup 兼 producer（tid0 分支）与 consumer，分支横跨活着的累加器
  寄存器，编译器插 warpgroup.arrive 导致潜在串行化。实测影响可接受（103T 已达标），
  CUTLASS 的解法是独立 producer warpgroup（warp specialization），列入下一步。
- 累加器写出走普通 global store（128 线程各 32 个 f32），无 TMA store / 无 swizzle
  ——epilogue 未优化，8192³ 下占比小，2048³ 下有优化空间。
- 无 OOB 处理：M/N/K 须为 64 的倍数（correctness/bench 尺寸均满足）。

## 5b: warp specialization + BM=128（`gemm_ws.cu`）

上面"已知问题"里的 C7520 在这里修掉，顺手把 tile 升到 BM=128。

### 结构（384 线程 = 3 warpgroup）

```
WG0 (tid 0..127)    producer：仅 tid0 跑独立 issue 循环（无累加器寄存器）
WG1/WG2 (128..383)  consumer：各跑 m64n64k16，分别管输出的行 0-63 / 64-127
```

- tile: **BM=128**, BN=BK=64；stage = A 16KB + B 8KB = 24KB（SW128）。
  两 consumer 共享同一份 B —— B 的 smem/带宽开销被两个 WG 摊薄，这正是大 tile
  的意义。屏障：`full[s]` count=1（producer 登记 24KB expect）、`empty[s]`
  count=256（两 consumer WG 全体 arrive）。producer 与 consumer **各跑各的循环**，
  只经 barrier 交互——不再有"issue 放迭代末尾防自锁"的约束（那是单 WG 兼职
  时的自我依赖）。
- **A 的 TMA box 是 {64,128}**（K×M），一次装载 128 行。SW128 布局按 8 行原子沿
  m 堆叠，行 64 起点 = 原子 8 = 偏移 8192（1024B 对齐，硬件从地址位恢复相位，
  SBO 不变）。consumer wg 的描述符 = `sA + wg*8192 + kk*32`——同一 stage 里
  两个 WG 各取各的 64 行半区，零拷贝。
- consumer 内循环与 Phase 5 完全相同（fence → 4×wgmma → commit → wait<1> →
  arrive empty），epilogue 行号 = `64*wg + m`。

### C7520 验证

| 内核 | ptxas 输出 |
|---|---|
| gemm_fused（单 WG 兼 producer） | 4× C7519 + **5× C7520**（串行化警告） |
| gemm_ws（producer 独立 WG） | 15× C7519（info）+ **0× C7520** |

producer WG 不持有累加器，divergent 路径不再横跨活着的 GMMA 寄存器窗口，
编译器无需插 warpgroup.arrive 兜底——警告消失。

### 数据（contended，min_ms）

正确性: 512×512×256, S∈{2,3,4} 全 PASS。

2048×2048×1024:

| stages | min_ms | TFLOPS |
|-------:|-------:|-------:|
| 2      | 0.091  | 94.4  |
| 3      | 0.074  | 116.2 |
| 4      | 0.073  | **117.6** |
| 6      | 0.091  | 94.6  |
| 8      | 0.091  | 93.9  |

8192×8192×8192:

| stages | min_ms | TFLOPS |
|-------:|-------:|-------:|
| 2      | 15.30  | 71.9  |
| 3      | 15.07  | 73.0  |
| 4      | 15.04  | **73.1** |
| 6      | 15.35  | 71.7  |

- **2048 尺寸 +13.5%**（117.6 vs 103.6）。8192³ 73.1T，微超 Phase 5 的 72.2T
  （该尺寸已 compute-bound，收益天花板就是张量核吞吐）。
- **最优 S 从 2 变成 4**：独立 producer 后，TMA issue 与 wgmma 不再共享一个
  warpgroup 的发射带宽，更深流水线才有意义；S=6 起掉头向下还是 occupancy
  （S=6 → 144KB smem → 1 CTA/SM，512 CTA / 78 SM 波数量化）。
- ncu 计数器（8192³ 最优配置，与 Phase 6 的 multicast 验证同批）：L2 读 sector
  0.772G ≈ 24.7GB，正落在 **BM=128 fetch 模型 24GB** 上（8192 CTA × (A 2MB +
  B 1MB)，BM=64 时是 32GB）——fetch 少 25%，行波数减半还顺带降 DRAM 读
  14.1→8.9GB（B 的跨波重取减少）。

### 记录

- producer 循环里 `j >= S` 才等 empty——前 S 个 stage 无条件发射（prologue），
  与 Phase 5 相同。
- 寄存器压力比想象小：`cuobjdump -res-usage` 实测 S=4 版本 **58 reg/线程**
  （384 线程 × 2 CTA = 44.5K reg < 64K，S=4 的 96KB×2=192KB smem 也放得下，
  双 CTA/SM 驻留无压力）——累加器经 wgmma 操作数走专用路径，普通寄存器
  配额没有被 32 个 f32 拖垮。
- M 须为 128 的倍数（BM=128）。

## 下一步

- epilogue: TMA store + swizzle 写出
- warp specialization × multicast 组合（producer WG 天然适合接管 cluster 的
  leader 发射职责）
- 真空复测（与 Phase 3 的 idle-watch 一并）
