add_executable(copy_compare copy_compare.cu)
target_include_directories(copy_compare PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
