cmake_minimum_required(VERSION 3.22)
project(cuda_mlp_model CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(cuda_mlp_model mlp_model.cuh src/mlp_model.cu src/host_mlp_model.cu main.cu)
#add_library(cuda_mlp_model SHARED mlp_model.cuh src/mlp_model.cu)


set_target_properties(cuda_mlp_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)




