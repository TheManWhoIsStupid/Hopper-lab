cmake_minimum_required(VERSION 3.18)

# Hopper 专属指令（TMA / wgmma / cluster）要求 sm_90a。
# CMake < 3.24 不认识 "90a" 架构值，因此关闭内置架构标志，手动传 -arch=sm_90a。
if(CMAKE_VERSION VERSION_LESS 3.24)
  set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()

project(hopper_lab LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -lineinfo: 给 ncu/ncfg 提供源码行号映射， profiling 必备
# 注意: 必须用 -gencode 形式。本机 nvcc 12.9 对 -arch=sm_90a 会静默降级成 compute_90
# （TMA 不受影响，但 wgmma/cluster 等 90a 专属指令会报 "not supported on .target sm_90"）。
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>
                    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_subdirectory(src/00_smoke)
add_subdirectory(src/01_memcpy)
add_subdirectory(src/02_pipeline)
add_subdirectory(src/03_compute)
add_subdirectory(src/04_wgmma)
add_subdirectory(src/05_gemm)
