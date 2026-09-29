add_executable(gemm_fused gemm_fused.cu)
target_include_directories(gemm_fused PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gemm_fused PRIVATE CUDA::cuda_driver)

add_executable(gemm_ws gemm_ws.cu)
target_include_directories(gemm_ws PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gemm_ws PRIVATE CUDA::cuda_driver)
