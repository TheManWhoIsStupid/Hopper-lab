// Phase 2: mbarrier 多级流水线 (producer-consumer)。
// 结构: S 个 stage 组成环形缓冲, 每 stage 一对 mbarrier:
//   full[s]  arrive count = 1   —— producer(tid0) arrive.expect_tx + TMA complete_tx 记账
//   empty[s] arrive count = 256 —— 全体消费者 arrive, producer 等它确认缓冲区可覆写
// S=1 时 TMA 与消费严格串行(基线); S>=2 时消费 stage i 的同时 TMA 装载 stage (i+1)%S。
// 奇偶跟踪: stage 的第 k 次使用对应 barrier 第 k 个 phase, parity = k & 1,
// 其中 k = i / S (i 为本 block 的全局迭代序号)。首轮 S 个 stage 天然空闲, empty 跳过等待。
//
// 消费刻意做成"搬运同量级": 每线程 128 次伪随机 smem 读(8x 数据复用, 模拟 GEMM
// 操作数重读), 这样 S=1 -> S=2 的重叠收益才看得见。
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "common/bench.h"
#include "common/checker.h"
#include "common/device_info.h"
#include "common/errors.h"
#include "common/tensor.h"
#include "common/timer.h"

namespace {

constexpr uint32_t kStageBytes = 16384;  // 16KB / stage (4096 fp32)
constexpr uint32_t kStageElems = kStageBytes / 4;
constexpr uint32_t kBlock = 256;
constexpr uint32_t kTiles = 16384;  // 16384 * 16KB = 256MB
constexpr uint32_t kConsumeReads = 128;  // 每线程每 stage 的 smem 读次数

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 自旋等待 barrier 上指定奇偶的 phase 完成（{} 作用域让 label 局部化）
__device__ __forceinline__ void wait_parity(const uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t}\n" ::"r"(smem_u32(bar)),
      "r"(parity));
}

// 流水线结构（关键！）: producer 必须领先消费者 S-1 个迭代发 TMA——
// 在迭代 i 的开头为 i+S-1 装载数据，消费 i 的同时后面 S-1 个 load 在飞。
// （反面教材见 git 历史: 若在同一迭代内 issue 后立即 wait，load 延迟完全暴露，
//  多级 stage 毫无收益。）
template <uint32_t S>
__global__ void __launch_bounds__(kBlock)
pipeline_kernel(const float* __restrict__ in, float* __restrict__ out,
                uint32_t num_tiles) {
  extern __shared__ __align__(128) char smem_raw[];
  char* stages = smem_raw;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_raw + S * kStageBytes);
  uint64_t* empty = full + S;

  const uint32_t tid = threadIdx.x;
  const uint32_t num_iters =
      (blockIdx.x < num_tiles)
          ? ((num_tiles - blockIdx.x + gridDim.x - 1) / gridDim.x)
          : 0;

  if (tid == 0) {
    for (uint32_t s = 0; s < S; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(
          smem_u32(&full[s])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(
                       smem_u32(&empty[s])),
                   "r"(kBlock));
    }
  }
  __syncthreads();

  // issue(j): 为第 j 个迭代装载 TMA 到 stage j%S
  auto issue = [&](uint32_t j) {
    if (j >= S) {  // 首轮 S 个 stage 天然空闲
      wait_parity(&empty[j % S], ((j / S) - 1) & 1);
    }
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
            smem_u32(&full[j % S])),
        "r"(kStageBytes));
    const float* src = in + size_t(j * gridDim.x + blockIdx.x) * kStageElems;
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];\n" ::"r"(smem_u32(stages + (j % S) * kStageBytes)),
        "l"(src), "r"(kStageBytes), "r"(smem_u32(&full[j % S])));
  };

  // prologue: 先把前 S-1 个 load 发出去（填管线）
  if (tid == 0) {
    for (uint32_t j = 0; j + 1 < S && j < num_iters; ++j) issue(j);
  }

  float acc = 0.f;
  for (uint32_t i = 0; i < num_iters; ++i) {
    if (tid == 0 && i + S - 1 < num_iters) {
      issue(i + S - 1);  // 保持管线满载
    }
    wait_parity(&full[i % S], (i / S) & 1);
    const float* stage =
        reinterpret_cast<const float*>(stages + (i % S) * kStageBytes);
#pragma unroll
    for (uint32_t k = 0; k < kConsumeReads; ++k) {
      // 跨线程步长 13(与 32 互素) -> 无 bank conflict
      acc += stage[(tid * 13 + k * 71) & (kStageElems - 1)];
    }
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(
        smem_u32(&empty[i % S])));
  }

  out[blockIdx.x * kBlock + tid] = acc;
}

template <uint32_t S>
void launch_config(const float* din, float* dout, uint32_t num_tiles, int grid) {
  const size_t smem = S * kStageBytes + S * 2 * sizeof(uint64_t);
  // 动态 smem 超 48KB 必须 opt-in（Hopper 上限 227KB）
  CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(pipeline_kernel<S>),
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem)));
  pipeline_kernel<S><<<grid, kBlock, smem>>>(din, dout, num_tiles);
  CUDA_CHECK_LAST();
}

void dispatch(uint32_t stages, const float* din, float* dout, uint32_t n, int grid) {
  switch (stages) {
    case 1: launch_config<1>(din, dout, n, grid); break;
    case 2: launch_config<2>(din, dout, n, grid); break;
    case 3: launch_config<3>(din, dout, n, grid); break;
    case 4: launch_config<4>(din, dout, n, grid); break;
    case 6: launch_config<6>(din, dout, n, grid); break;
    default: std::fprintf(stderr, "未支持的 stage 数: %u\n", stages); std::exit(1);
  }
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  const size_t n_floats = size_t(kTiles) * kStageElems;
  auto hin = hopper::make_random_vector(n_floats, 23);
  hopper::DeviceBuffer<float> din(n_floats);
  din.upload(hin);
  hopper::DeviceBuffer<float> dout(kBlock * 1024);  // grid 上限 78*4=312 blocks

  // ---- 正确性 (grid=156, 每个 S 各验一遍) ----
  // 流水线不改变语义: acc = 按同顺序累加同样的值, 应 bit-exact。
  // 注意 host 端必须按 device 的扁平顺序逐项累加（不能先算每 tile 部分和再相加，
  // 浮点加法不满足结合律）。
  const int grid_chk = caps.sm_count * 2;
  std::vector<float> expected(size_t(grid_chk) * kBlock, 0.f);
  for (int b = 0; b < grid_chk; ++b) {
    for (uint32_t tid = 0; tid < kBlock; ++tid) {
      float acc = 0.f;
      for (uint32_t t = b; t < kTiles; t += grid_chk) {
        for (uint32_t k = 0; k < kConsumeReads; ++k) {
          acc += hin[size_t(t) * kStageElems + ((tid * 13 + k * 71) & (kStageElems - 1))];
        }
      }
      expected[size_t(b) * kBlock + tid] = acc;
    }
  }
  for (uint32_t s : {1u, 2u, 3u, 4u, 6u}) {
    dispatch(s, din.get(), dout.get(), kTiles, grid_chk);
    auto raw = dout.download();  // 缓冲区按最大 grid 分配，只比对本次写入的前缀
    std::vector<float> got(raw.begin(), raw.begin() + expected.size());
    auto rep = hopper::check_close(got, expected, 0.0, 0.0);
    char tag[32];
    std::snprintf(tag, sizeof(tag), "stages=%u", s);
    hopper::print_report(rep, tag);
    if (!rep.pass) return 1;
  }

  // ---- 基准 ----
  const double bytes_per_pass = double(kTiles) * kStageBytes;  // 只计 DRAM 读
  const double peak = hopper::theoretical_bw_gbps(caps);
  std::printf(
      "\n[BENCH ] 流水线 stage 数扫描: 16KB tile, 消费=128次smem读/线程(8x复用)\n"
      "         数据搬运量/pass = %.0f MB (DRAM 读, 消费只在 smem)\n\n",
      bytes_per_pass / (1024.0 * 1024.0));
  std::printf("  %-7s %6s %8s %12s %10s %8s\n", "stages", "grid", "blk/SM",
              "min_ms", "GB/s", "%peak");

  std::filesystem::create_directories("results");
  hopper::CsvWriter csv("results/pipeline.csv",
                        {"stages", "grid", "blocks_per_sm", "min_ms", "mean_ms",
                         "gbps", "pct_peak"});
  for (int bpsm : {1, 2, 4}) {
    const int grid = caps.sm_count * bpsm;
    for (uint32_t s : {1u, 2u, 3u, 4u, 6u}) {
      auto st = hopper::time_reps(5, 30, [&] { dispatch(s, din.get(), dout.get(), kTiles, grid); });
      double bw = hopper::bandwidth_gbps(bytes_per_pass, st.min_ms);
      std::printf("  %-7u %6d %8d %12.4f %12.1f %7.1f%%\n", s, grid, bpsm,
                  st.min_ms, bw, bw / peak * 100.0);
      csv.row({std::to_string(s), std::to_string(grid), std::to_string(bpsm),
               std::to_string(st.min_ms), std::to_string(st.mean_ms),
               std::to_string(bw), std::to_string(bw / peak * 100.0)});
    }
  }
  return 0;
}
