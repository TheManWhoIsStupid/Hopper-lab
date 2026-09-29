add_executable(smoke_vector_add vector_add.cu)
target_include_directories(smoke_vector_add PRIVATE ${PROJECT_SOURCE_DIR}/include)
