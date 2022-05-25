cmake_minimum_required(VERSION 3.22)
project(GD_cpp_linear_model)

set(CMAKE_CXX_STANDARD 23)
include_directories("${PROJECT_SOURCE_DIR}\\Eigen")

add_executable(GD_cpp_linear_model main.cpp linear_model.h src/linear_model.cpp)
target_link_libraries(GD_cpp_linear_model ${PROJECT_SOURCE_DIR}\\Eigen)