cmake_minimum_required(VERSION 3.21)
project(GameDetection)

set(CMAKE_CXX_STANDARD 23)
include_directories("${PROJECT_SOURCE_DIR}\\Eigen")
add_executable(GameDetection main.cpp
        inc/linear_model.h
        src/linear_model.cpp
        inc/mlp_model.h
        src/mlp_model.cpp
        )

target_link_libraries(GameDetection ${PROJECT_SOURCE_DIR}\\Eigen)