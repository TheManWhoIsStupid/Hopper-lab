add_executable(copy_compare copy_compare.cu)
target_include_directories(copy_compare PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# TMA 2D 需要 driver API 的 cuTensorMapEncodeTiled
add_executable(tma2d_swizzle tma2d_swizzle.cu)
target_include_directories(tma2d_swizzle PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(tma2d_swizzle PRIVATE CUDA::cuda_driver)
