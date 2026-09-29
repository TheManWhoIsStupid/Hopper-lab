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

## 下一步

- warp specialization：producer 独立 warpgroup（修 C7520），consumer 2×warpgroup
  BM=128（m64n64k16 ×2 拼）
- epilogue: TMA store + swizzle 写出
- cluster + DSMEM（Phase 6）: 跨 CTA 共享 A tile，省一半 smem 流量
- 真空复测（与 Phase 3 的 idle-watch 一并）
