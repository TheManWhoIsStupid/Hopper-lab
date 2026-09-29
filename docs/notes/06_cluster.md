# Phase 6: Thread Block Cluster + DSMEM + TMA multicast

代码: `src/06_cluster/cluster_dsmem.cu`（6a）、`src/06_cluster/tma_multicast.cu`（6b），
原语封装 `include/common/cluster.cuh`。

## 机制速查

- **launch**: `__global__ void __cluster_dims__(2,1,1) kernel(...)`，grid 对应维须整除。
  cluster 内 CTA 保证同驻（同 GPC 不同 SM），可互访 shared memory。
- **rank**: `mov.u32 r, %cluster_ctarank` —— 本 CTA 在 cluster 内编号。
- **DSMEM 寻址**: `mapa.shared::cluster.u32 dst, local_smem_addr, target_rank`
  把本 CTA 的 shared 地址映射到 target CTA 的同偏移位置；之后用
  `ld/st.shared::cluster`（`.v4.u32` 可用）读写。
- **cluster barrier**: `barrier.cluster.arrive.aligned` / `wait.aligned`，默认带
  cluster 作用域 release/acquire——arrive 前的写在 wait 后对全 cluster 可见。
- **mbarrier.init 的可见性**: 跨 CTA 使用（远程 arrive / TMA 记账）前需要
  `fence.mbarrier_init.release.cluster` + cluster barrier。
- **远程 arrive**: `mbarrier.arrive.shared::cluster.b64 _, [mapa后的地址]`——
  别的 CTA 可以来 arrive 我的 barrier（cluster 流水线的释放同步就靠它）。
- **TMA multicast**: `cp.async.bulk.tensor.2d.shared::cluster.global
  .mbarrier::complete_tx::bytes.multicast::cluster [dst],[tmap,{c0,c1}],[bar], ctaMask`
  （指令串对齐 cute/arch/copy_sm90_tma.hpp，其 `.tile` 可省）。

## 6a: DSMEM 交换 + 远程 mbarrier（全 PASS）

156 CTA = 78 个 2-CTA cluster（每 SM 2 CTA），交换 16KB 图案：远程读、远程写、
邻居 arrive 我的 mbarrier 我等相位——mismatch=0。

读带宽（uint4 连读，clock64 计时，contended）:

| 模式 | min (GB/s/CTA) | mean (GB/s/CTA) |
|---|---:|---:|
| 本地 `ld.shared.v4` | 63.9 | 64.8 |
| 远程 `ld.shared::cluster.v4` | 7.5 | 7.8 |

**DSMEM 细粒度读 ≈ 本地的 1/8**。结论：跨 CTA 数据搬运不要用零散远程 ld，
要么 TMA multicast（硬件块传输），要么 `st.async`/bulk copy；DSMEM 远程读只适合
低频控制类访问。

## 6b: TMA multicast 记账语义（本阶段最重要的实锤）

### 发射方实验

| 发射方式 | 结果 |
|---|---|
| 仅 rank0（leader）发射，两 CTA 各自 `arrive_expect_tx(8KB)` | **PASS** |
| 全体 CTA 都发射 | **挂死**（`--all-issue` 可复现） |

⇒ **一条 multicast 指令给 mask 内每个目的 CTA 的同偏移 mbarrier 各记一次
complete_tx**。所以：每个接收 CTA 对自己的 full barrier 登记一次期望字节数，
指令只由一个 CTA 发射；全体发射 = 每 barrier 双重记账 → 相位错乱 → 流水线死锁。
（mbarrier 的 tx 记账是相位敏感的：超额 complete_tx 会打乱后续相位的奇偶，
表现为 wait_parity 永假自旋。）

注意本实验的简化：empty barrier 是每 CTA 私有的（消费者无计算即刻释放）。
生产级流水线里 rank1 的 smem 同样被 multicast 覆写，释放同步必须 cluster 化：
全体消费者**远程 arrive leader 的同一个 empty barrier**（count = 消费者×cluster），
仅 leader 等待并发射——这正是远程 arrive 存在的意义。

### 带宽（156 CTA 读同一条 2MB L2 驻留 strip，S=4 流水线）

| mode | min_ms | 接收口径 (GB/s) | fetch 口径 (GB/s) |
|---|---:|---:|---:|
| unicast | 0.067 | 4908 | 4908 |
| mcast   | 0.069 | 4764 | **2382** |

- 接收带宽持平（差 3%，噪声级）而 **fetch 侧流量减半**。
- 解释: 此配置下传送路径是**延迟/在飞量限制**（4 stage × 8KB × 156 CTA ≈ 5MB
  在飞，约折算 5-7TB/s 上限），未触及 L2→SMEM 带宽墙，所以 multicast 不会让
  接收翻倍——它省的是 fetch 侧（L2 查找/读带宽）。GEMM 中 L2 带宽要同时伺候
  A/B 两种 tile 和其他 CTA，这正是 multicast 的价值所在；想观察到接收端翻倍
  需要更深流水或更大 tile 把 L2 打满。

## 踩坑

1. **`barrier.cluster` 的 `.aligned` 不能埋进线程分歧**：tid0 分支内执行
   `barrier.cluster.arrive`（warp0 内 lane0 与其余 lane 走不同路径）→ 挂死。
   cluster barrier 必须全 CTA 收敛执行；分歧部分只允许放非 cluster-barrier 的
   操作（如远程 arrive）。
2. （沿用 Phase 5）多级流水的 issue 放迭代末尾；本阶段新增：multicast 的
   expect_tx 在每个接收 CTA 各做一次、发射只在一处。

## 下一步

- GEMM 集成: 2-CTA cluster 沿 N 维配对（同 bm 共享 A tile），A 走 multicast、
  B 走 unicast；empty barrier 按"远程 arrive leader"的生产级做法 cluster 化
- warp specialization（producer 独立 warpgroup，顺带修 Phase 5 的 C7520）
- vacuum 复测（idle-watch cron 待机中）
