#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include "common/errors.h"

namespace hopper {

// Hopper (sm_90) 研究关心的设备能力子集
struct DeviceCaps {
  char name[256];
  int cc_major, cc_minor;
  int sm_count;
  int clock_khz;             // SM 核心频率
  size_t total_mem;          // 显存总量 (bytes)
  size_t l2_bytes;           // L2 cache 大小
  int smem_per_block_default;  // 动态 shared memory 默认上限 (KB)
  int smem_per_block_optin;    // opt-in (cudaFuncAttributeMaxDynamicSharedMemorySize) 后上限 (KB)
  int smem_per_sm;             // 每 SM 的 shared memory 容量 (KB)
  int mem_clock_khz;
  int mem_bus_bits;
  int cluster_launch;          // 是否支持 thread block cluster
};

inline DeviceCaps query_device(int device_id = 0) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  DeviceCaps c{};
  snprintf(c.name, sizeof(c.name), "%s", prop.name);
  c.cc_major = prop.major;
  c.cc_minor = prop.minor;
  c.sm_count = prop.multiProcessorCount;
  c.clock_khz = prop.clockRate;
  c.total_mem = prop.totalGlobalMem;
  c.l2_bytes = prop.l2CacheSize;
  c.smem_per_block_default = static_cast<int>(prop.sharedMemPerBlock / 1024);
  c.smem_per_block_optin =
      static_cast<int>(prop.sharedMemPerBlockOptin / 1024);
  c.smem_per_sm = static_cast<int>(prop.sharedMemPerMultiprocessor / 1024);
  c.mem_clock_khz = prop.memoryClockRate;
  c.mem_bus_bits = prop.memoryBusWidth;
  c.cluster_launch = prop.clusterLaunch;
  return c;
}

// 理论显存带宽 (GB/s)。按 DDR 双倍数据率估算，仅作参考基准。
inline double theoretical_bw_gbps(const DeviceCaps& c) {
  double bytes_per_s = 2.0 * c.mem_clock_khz * 1000.0 * (c.mem_bus_bits / 8.0);
  return bytes_per_s / 1e9;
}

inline void print_device_caps(const DeviceCaps& c) {
  std::printf("=== Device ===\n");
  std::printf("  name                  : %s (cc %d.%d)\n", c.name, c.cc_major,
              c.cc_minor);
  std::printf("  SM count              : %d @ %.0f MHz\n", c.sm_count,
              c.clock_khz / 1000.0);
  std::printf("  global memory         : %.1f GB, L2 %zu KB\n",
              c.total_mem / (1024.0 * 1024.0 * 1024.0), c.l2_bytes / 1024);
  std::printf("  shared mem / block    : %d KB (default) / %d KB (optin), per SM %d KB\n",
              c.smem_per_block_default, c.smem_per_block_optin, c.smem_per_sm);
  std::printf("  memory bus            : %d-bit @ %.2f GHz (~%.0f GB/s peak)\n",
              c.mem_bus_bits, c.mem_clock_khz / 1e6, theoretical_bw_gbps(c));
  std::printf("  cluster launch        : %s\n", c.cluster_launch ? "yes" : "no");
}

// 本仓库只面向 Hopper (sm_90+)，其他架构直接退出给出可读信息
inline void require_hopper(const DeviceCaps& c) {
  if (c.cc_major < 9) {
    std::fprintf(stderr,
                 "\n此仓库针对 Hopper (sm_90) 研究，当前设备 %s (cc %d.%d) 不满足要求\n",
                 c.name, c.cc_major, c.cc_minor);
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace hopper
