add_executable(peak_flops peak_flops.cu)
target_include_directories(peak_flops PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(cublas_gemm cublas_gemm.cu)
target_include_directories(cublas_gemm PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cublas_gemm PRIVATE CUDA::cublas CUDA::cublasLt)
