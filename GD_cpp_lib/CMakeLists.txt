cmake_minimum_required(VERSION 3.22)
project(mlp_model)

set(CMAKE_CXX_STANDARD 23)
include_directories("${PROJECT_SOURCE_DIR}\\Eigen")

add_executable(mlp_model main.cpp src/mlp_modelV2.cpp inc/mlp_modelV2.h)
#add_library(mlp_model SHARED src/mlp_modelV2.cpp inc/mlp_modelV2.h)

target_link_libraries(mlp_model ${PROJECT_SOURCE_DIR}\\Eigen)