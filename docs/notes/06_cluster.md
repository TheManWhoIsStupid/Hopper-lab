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

## 6c: multicast GEMM（`gemm_cluster.cu`）

Phase 5 流水线 + cluster 数据复用：`__cluster_dims__(2,1,1)` 让 blockIdx.x 相邻
两 CTA 成对（同 bm 不同 bn，共享 A tile）。A 走 multicast（leader 发射一条，
两 CTA 各收一份），B 各自 unicast——**fetch 侧 A 流量减半**。

屏障拓扑（6b 语义的直接应用）:

| barrier | 位置 | count | 用途 |
|---|---|---|---|
| `full[s]` | 每 CTA | 1 | 各自 tid0 登记 16KB expect |
| `empty_a[s]` | 仅 leader | 256 | 两 CTA 全体消费者 arrive（rank1 远程）——multicast 覆写两边 A 区，释放须双 CTA 确认；try_wait 只能本地等 ⇒ 放 leader |
| `empty_b[s]` | 每 CTA | 128 | 自己的 B 区自己管 |
| `armed[s]` | 仅 leader | 1 | rank1 登记 expect 后远程 arrive，rank0 见到才发射 multicast |

`armed` 的必要性：mbarrier 的 tx 记账是**相位敏感**的。若 rank0 的 multicast
记账先于 rank1 的 expect 登记落地，credit 会记到上一个（已完成的）相位——
该 stage 永远等不满 → 挂死。armed 用一跳远程 arrive 把"登记完成"显式通知
发射方，窗口彻底闭合（CUTLASS 用 cluster transaction barrier 干同样的事）。

正确性: 512×512×256，S∈{2,3,4} 各 ×5 重复——全 PASS。

| 尺寸 | 最优 | Phase 5 对照 |
|---|---|---|
| 2048×2048×1024 | 95.9T (S=2) | 103.6T（**-7%**） |
| 8192³ | 72.5T (S=3) | 72.2T（持平） |

诚实结论：当前条件下 multicast **没有可测的 wall-clock 收益**，2048 尺寸还有
个位数回退。归因：(1) 8192³ 是 compute-bound（72T 已打平 cuBLAS），fetch 减半
省的是 L2 带宽，只有 ncu 计数器（`lts__t_bytes`）能验证；(2) 每 stage 多了
4 次 barrier 交互（远程 arrive + armed 等待），关键路径加了一跳；(3) 争用下
相对排序本就不可信（Phase 3 教训）。multicast 的真实收益场景：更大 tile/更宽
cluster（2×2）、L2 压力大的形状、或与 warp specialization 组合后 compute 更快
时——留待后续。

## 踩坑（6c 新增）

- **cluster 内核收尾必须 cluster_sync**：最后一批远程 arrive（rank1→leader 的
  empty_a/armed）可能还在互连上飞；先退出的 CTA smem 被回收，迟到 arrive 打到
  已释放地址 = **ULF**（8192³ S=3 实测复现，512 正确性 15 连全过也拦不住——
  纯时序竞态）。`barrier.cluster` 的 release 语义保证先于它的远程访存已落地，
  结尾加一道即可闭合。

## 6c 补充: ncu 计数器验证 multicast fetch 减半

6b 的"fetch 减半"是从带宽口径推的（recv 持平 + 每指令两份交付 ⇒ fetch=recv/2），
6c 完成后用 ncu 直接采 L2 计数器坐实（`/usr/local/NVIDIA-Nsight-Compute-2025.4/ncu`，
`--launch-skip` 精确切到 8192³ 最优 S 的那个 launch；contended GPU 上采的计数器
同样带背景噪声，但三组数据能互相印证）:

| kernel | tile | L2 读 sectors (`lts__t_sectors_op_read`) | 折算 | fetch 模型 |
|---|---|---:|---:|---:|
| gemm_fused S=4 | BM=64 | 1.283G | 41.1GB | 32GB |
| gemm_cluster S=3 | BM=64, A mcast | 1.061G | 34.0GB | 24GB |
| gemm_ws S=4 | BM=128 | 0.772G | 24.7GB | 24GB |

- **cluster vs fused（唯一差异是 A 走 multicast）: L2 读 -7.1GB / -17.3%**，
  与"A 流量减半"模型的 8GB 基本吻合（89%）——multicast 省 fetch 在计数器
  层面成立。不足额部分归因于共卡背景流量（sglang 同时在打 L2）。
- gemm_ws（BM=128，无 multicast）精确落在 24GB 模型上，反证 fused 的 +9GB
  超额不是系统常量背景，而与 BM=64 的访问模式/共卡瞬时负载有关——真空
  复测时值得重采一组。
- `dram__bytes_read`: 14.1 / 15.3 / 8.9 GB——方向对（ws 行波减半 ⇒ B 的跨波
  重取减少）但绝对值受 ncu 清缓存 + 共卡干扰大，只作参考。

## 下一步

- 2×2 cluster / 与 warp specialization 组合——multicast 收益要配合更大的
  数据复用面（gemm_ws 的 producer WG 天然适合接管 leader 发射职责）
- vacuum 复测时重采一组干净的 lts/dram 计数器（顺带补 Phase 1 的
  dram__bytes 验证）
