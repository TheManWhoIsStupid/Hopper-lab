add_executable(wgmma_fp8 wgmma_fp8.cu)
target_include_directories(wgmma_fp8 PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(gemm_fp8 gemm_fp8.cu)
target_include_directories(gemm_fp8 PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gemm_fp8 PRIVATE CUDA::cuda_driver)
