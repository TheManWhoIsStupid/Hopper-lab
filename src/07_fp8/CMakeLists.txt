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

add_executable(gemm_fp8_cluster gemm_fp8_cluster.cu)
target_include_directories(gemm_fp8_cluster PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gemm_fp8_cluster PRIVATE CUDA::cuda_driver)

add_executable(gemm_fp8_2x2 gemm_fp8_2x2.cu)
target_include_directories(gemm_fp8_2x2 PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gemm_fp8_2x2 PRIVATE CUDA::cuda_driver)

add_executable(gemm_fp8_2x2_uni gemm_fp8_2x2_uni.cu)
target_include_directories(gemm_fp8_2x2_uni PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gemm_fp8_2x2_uni PRIVATE CUDA::cuda_driver)
