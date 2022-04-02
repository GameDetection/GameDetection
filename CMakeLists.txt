cmake_minimum_required(VERSION 3.21)
project(GameDetection)

set(CMAKE_CXX_STANDARD 23)
include_directories("${PROJECT_SOURCE_DIR}\\Eigen")
add_executable(GameDetection main.cpp
        inc/model.h
        src/model.cpp
        )

target_link_libraries(GameDetection ${PROJECT_SOURCE_DIR}\\Eigen)