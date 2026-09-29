add_executable(pipeline_stages pipeline_stages.cu)
target_include_directories(pipeline_stages PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
