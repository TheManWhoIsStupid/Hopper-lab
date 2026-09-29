add_executable(wgmma_basic wgmma_basic.cu)
target_include_directories(wgmma_basic PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(wgmma_sw128 wgmma_swizzle128.cu)
target_include_directories(wgmma_sw128 PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
