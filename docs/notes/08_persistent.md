# Phase 8: Persistent Kernel

代码: `src/08_persistent/gemm_fp8_persist.cu`。8192³ 的 tile 数（8192）≫
SM 数（78），非 persistent 启动下每个 CTA 一个 tile、跑完即换——波边界上
流水线整个排空重启。persistent 让 CTA 驻留循环领 tile，两个红利预期：
(1) 流水线跨 tile 不冷；(2) tile 调度权到手，可以用 group-M swizzle 把
行波 DRAM 重取消掉（7b 实测 8192³ DRAM 读 4.28GB，其中 ~4GB 是 B 行波）。

## 8a 设计：7b ws 结构 + 三个 persistent 专属改动

计算通路与 7b 完全相同（1 producer WG + 2 consumer WG，fp8 BK=128，
mbarrier S 级环）。grid 改为 1D 驻留集 = `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
× SM 数（上限 total_tiles；S≥4 时 smem ≥128KB ⇒ 1 CTA/SM，grid=78）。
CTA `for (tile = blockIdx.x; tile < total; tile += gridDim.x)` 静态轮转领 tile。

persistent 化的三处结构性改动（都有明确的"为什么 7b 不需要"）：

1. **kblock→stage 环跨 tile 连续**：`j` 是全 kernel 计数器（不是每 tile 归零），
   tile t 的尾段与 tile t+1 的预取由 full/empty 屏障自然重叠。7b 每 CTA 只算
   一个 tile，环绕不发生。
2. **tile 末尾补释放尾级 stage**：`wgmma_wait_group<0>()` 之后对
   `empty[(j-1)%S]` 补一次 arrive。7b 里最后一级不等释放直接退出（kernel
   结束 = 全体资源回收）；persistent 里环要绕回，漏放这一级 = 下一圈
   producer 在该 stage 上死等——挂点在恰好 S 个 kblock 之后，确定性死锁。
3. **epilogue 暂存专用区（sEp 2×16KB）**：7b/5c 复用 sA 头两级的做法在
   persistent 下是竞态——consumer 写 epilogue 时 producer 已在往 sA 预取
   下一 tile。副产品：跨 WG 的 `named_barrier(3)` 不再需要（各 WG 只用
   自己的 16KB 半区，`named_barrier(1+wg,128)` 即可）。代价 32KB smem。

tile 调度（CUTLASS TileScheduler 同款 group-M swizzle）：

```
g = tile / (G*num_bn);  gsize = min(num_bm - g*G, G);  t = tile - g*(G*num_bn)
bm = g*G + t % gsize;   bn = t / gsize
```

每组 G 个 bm 行，组内先跨行后沿列。并发窗口（78 CTA）覆盖 G × (78/G × G…)
紧凑矩形，A/B 双双落 L2。**G=1 严格退化为行优先 x-fastest**——与 CUDA
自然波序同构，这是与 7b 的公平对照组。

DRAM 读模型（8192³，B 全遍历 × 组数 + A compulsory）：
`DRAM = 4096/G + 64 MB`。G=1: 4.16GB（复现 7b 行波）；G=8: 576MB；
G=16: 320MB。L2 读下限 = compulsory 12.6GB（8192 tile × 64 kblock × 24KB）。

## 正确性

512×512×256（单波）+ **2048×2048×256（多波**，检验跨 tile 环 + 尾释放 +
暂存覆写排空），S∈{2,4,8} × G∈{1,8} × 3 rep 全 PASS——一次通过。
正确性矩阵专门覆盖 G=8（swizzle 映射）与多波（persistent 语义），不是摆设。

## 数据（同日满载，min_ms；7b 同日复跑做对照）

### wall-clock

| 形状 | persist 最优 | 7b 同日 | Δ |
|---|---|---:|---|
| 2048³×1024 | 200.3T (S=6 G=8) | 208.4T (tma S=4) | **-3.9%** |
| 8192³ | 174.5T (S=8 G=1) | 174.59T (tma S=4) | **持平** |

S 维度（2048）：S=2 181 → S=4 194 → S=6/8 ~198-200——persistent 下更深
流水线单调受益（7b 的 S=6 反而崩到 173.6T：smem 144KB 挤掉第二 CTA；
persist 反正全程 1 CTA/SM，没有可丢的占用）。G 维度 wall-clock 上 **无效**
（±0.5%）。

### ncu 计数器（8192³，同日串行 = 半净条件）

| kernel | DRAM 读 | L2 读 | 时间 | tensor pipe |
|---|---:|---:|---:|---:|
| 7b S=4 tma | 4.14GB | 14.4GB | 4.27ms | 24.54% |
| persist S=8 **G=1** | 3.59GB | 14.4GB | 4.25ms | 24.72% |
| persist S=8 **G=8** | **605MB** | 12.2GB | 4.25ms | 24.76% |
| persist S=8 **G=16** | **361MB** | 12.1GB | 4.25ms | 24.76% |

（launch 计数：本 binary 正确性 36 发 + 2048 bench 4S×4G×13=208 发 +
8192 3S×3G×8=72 发；S=8 G=1/8/16 的首个 timed rep = 295/303/311。

### 三个实锤

1. **DRAM 减量全额兑现，机制精确成立**。G=8: 605MB（模型 576MB，95%）；
   G=16: 361MB（模型 320MB，89%）；G=1: 3.59GB（模型 4.16GB，缺口 =
   L2 跨波残留少量 B）。L2 读同步收敛到 compulsory 下限 12.6GB（G≥8 达
   97%）——swizzle 后 B 的重访全部命中 L2，这是"调度权换流量"的直接证据。
   persistent + group-M 作为**流量工具**完全成立（-85%~-91% DRAM）。
2. **流量依然不是时间的货币（满载）**。-85% DRAM 买回 0.00ms（4.27→4.25，
   噪声内），tensor pipe 四行全部 ~24.7%。与 7c（multicast -10% L2 不兑
   现）、7d（-43% L2 兑现但被拓扑税吃掉）合流成同一条定律：**共卡满载争用
   是 SM 周期型，不是访存带宽型**——自家流量砍到 1/6，时间一步不动。
   流量工具的价值面收窄到：真空/带宽硬约束场景、功耗、以及（见 3）更大的
   矩阵。
3. **persistent ≠ 免费加速：小形状暴露静态均衡短板**。8192³（105
   tiles/CTA，不均衡摊薄到 1%）严格持平；2048³（512 tiles / 78 CTA =
   6.56，静态轮转让 44 个 CTA 跑 7 tile、34 个跑 6 ⇒ makespan +7% 理论，
   实测 -3.9%，动态硬件调度吃掉约一半差距）。修法明确：全局原子 tile
   队列（work stealing，CUTLASS dynamic scheduler 路线）→ 8b。
   顺带：persistent 的"流水线不冷"红利在满载下不可见（满载本来就慢在
   争用，不在波边界）——真空下 2048 是它的主战场，待 idle-watch 复测。

## 机制速查 / 坑

- **尾级释放是 persistent 环的生死线**：漏放 `empty[(j-1)%S]` 的挂点在
  下一 tile 的第 S 个 kblock（producer 等第 S+1 次覆写）——确定性死锁，
  且 correctness 的小 K（256 = 2 kblock）**测不出来**（环没绕回 S 圈），
  必须 multi-wave 大 K 用例覆盖（本组 2048×2048×256 只验了语义一半，
  K=8192 的 bench 本身就是另一半——首跑若挂会挂在 bench 而非 CHECK）。
- **专用 epilogue 暂存 vs 复用 sA**：persistent 下必须专用（producer 预取
  竞态），但注意 `tma_store_wait_read<0>()` 仍是每 tile 一次的**关键路径
  串行点**（暂存被 TMA 引擎读出前不能覆写）。更优做法是 sEp 双缓冲 +
  commit group 跨 tile 排空（本版未做，8b 候选）。
- ncu 计 launch 数时 bench 配置数要按**本 binary 的实际 sweep 表**数，
  不能沿用旧 binary 的（7b 2048 是 8 配置不是 4，本次踩过）。

## 下一步

- **真空复测加一行**：persist 8192³ G=8 vs 7b——若真空下 7b 的 260T 卡在
  DRAM 行波（4.1GB/4.25ms ≈ 0.97TB/s 自家口径），G=8 应显出 >260T；
  若仍 260T 附近，则快窗也是 compute-bound，"流量工具论"再 +1 票。
  （idle-watch cron 待机中。）
- 8b: 原子 tile 队列消 2048 静态不均（预期 +4% 拿回 7b 差距并反超）；
  sEp 双缓冲消每 tile store 排空。
- BN=128 大 tile（m64n128，64 累加器/WG）——与 persistent 天然同框
  （persistent 载体已在手，大 tile 直接往上叠）。
- e5m2 / per-tensor scale 量化策略（真实输入范围 ≠ [-1,1]）。
