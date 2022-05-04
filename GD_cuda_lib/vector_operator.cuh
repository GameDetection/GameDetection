//
// Created by Carmo on 03/05/2022.
//
#include "Eigen/Core"

#ifndef VECTOR_OPERATOR_VECTOR_OPERATOR_CUH
#define VECTOR_OPERATOR_VECTOR_OPERATOR_CUH

__global__ void vector_multiply(Eigen::VectorXf a, Eigen::VectorXf b, Eigen::VectorXf c);
__global__ void scalar_vector_multiply(Eigen::VectorXf a, float b, Eigen::VectorXf c);

__global__ void vector_add(Eigen::VectorXf a, Eigen::VectorXf b, Eigen::VectorXf c);
__global__ void scalar_vector_add(Eigen::VectorXf a, float b, Eigen::VectorXf c);

__global__ void vector_subtract(Eigen::VectorXf a, Eigen::VectorXf b, Eigen::VectorXf c);


__global__ void vector_square(Eigen::VectorXf a, Eigen::VectorXf c);

#endif VECTOR_OPERATOR_VECTOR_OPERATOR_CUH
