cmake_minimum_required(VERSION 3.22)
project(cuda_mlp_model CUDA)

set(CMAKE_CUDA_STANDARD 23)

include_directories("${PROJECT_SOURCE_DIR}\\Eigen")

add_executable(cuda_mlp_model main.cu mlp_model.cuh src/mlp_model.cu)
#add_library(vector_operator SHARED main.cu src/vector_operator.cu vector_operator.cuh mlp_model.cuh src/mlp_model.cu)

set_target_properties(cuda_mlp_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuda_mlp_model ${PROJECT_SOURCE_DIR}\\Eigen)





