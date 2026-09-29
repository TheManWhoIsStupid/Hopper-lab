add_executable(gemm_fp8_persist gemm_fp8_persist.cu)
target_include_directories(gemm_fp8_persist PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gemm_fp8_persist PRIVATE CUDA::cuda_driver)

add_executable(gemm_fp8_bn128 gemm_fp8_bn128.cu)
target_include_directories(gemm_fp8_bn128 PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gemm_fp8_bn128 PRIVATE CUDA::cuda_driver)
