add_executable(cluster_dsmem cluster_dsmem.cu)
target_include_directories(cluster_dsmem PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(tma_multicast tma_multicast.cu)
target_include_directories(tma_multicast PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(tma_multicast PRIVATE CUDA::cuda_driver)

add_executable(gemm_cluster gemm_cluster.cu)
target_include_directories(gemm_cluster PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gemm_cluster PRIVATE CUDA::cuda_driver)
