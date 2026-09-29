cmake_minimum_required(VERSION 3.18)

# Hopper 专属指令（TMA / wgmma / cluster）要求 sm_90a。
# CMake < 3.24 不认识 "90a" 架构值，因此关闭内置架构标志，手动传 -arch=sm_90a。
if(CMAKE_VERSION VERSION_LESS 3.24)
  set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()

project(hopper_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -lineinfo: 给 ncu/ncfg 提供源码行号映射， profiling 必备
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90a>
                    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_subdirectory(src/00_smoke)
add_subdirectory(src/01_memcpy)
