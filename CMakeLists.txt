cmake_minimum_required(VERSION 3.21)
project(mlp_model)

set(CMAKE_CXX_STANDARD 23)
include_directories("${PROJECT_SOURCE_DIR}\\Eigen")
add_executable(mlp_model main.cpp
        inc/linear_model.h
        src/linear_model.cpp
        inc/mlp_model.h
        src/mlp_model.cpp
        )

target_link_libraries(mlp_model ${PROJECT_SOURCE_DIR}\\Eigen)

#cmake_minimum_required(VERSION 3.21)
#project(mlp_model)
#
#set(CMAKE_CXX_STANDARD 23)
#include_directories("${PROJECT_SOURCE_DIR}\\Eigen")
#
#add_library(mlp_model SHARED inc/mlp_model.h src/mlp_model.cpp)
#
#target_link_libraries(mlp_model ${PROJECT_SOURCE_DIR}\\Eigen)