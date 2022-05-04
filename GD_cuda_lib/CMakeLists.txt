cmake_minimum_required(VERSION 3.22)
project(vector_operator CUDA)

set(CMAKE_CUDA_STANDARD 23)

include_directories("${PROJECT_SOURCE_DIR}\\Eigen")

add_executable(vector_operator main.cu src/vector_operator.cu vector_operator.cuh mlp_model.cuh src/mlp_model.cu)
#add_library(vector_operator SHARED main.cu src/vector_operator.cu vector_operator.cuh mlp_model.cuh src/mlp_model.cu)

set_target_properties(vector_operator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(vector_operator ${PROJECT_SOURCE_DIR}\\Eigen)





