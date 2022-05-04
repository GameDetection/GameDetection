//
// Created by Carmo on 03/05/2022.
//
#include <Eigen>


//this function multiplies a Eigen vector by a scalar
__global__ void scalar_vector_multiply(Eigen::VectorXf a, float b, Eigen::VectorXf c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.size()) {
        c(i) = a(i) * b;
    }
}

//This function multiplies two Eigen vectors and stores the result in a third vector
__global__ void vector_multiply(Eigen::VectorXf a, Eigen::VectorXf b, Eigen::VectorXf c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.size()) {
        c(i) = a(i) * b(i);
    }
}

//This function adds two Eigen vectors and stores the result in a third vector
__global__ void vector_add(Eigen::VectorXf a, Eigen::VectorXf b, Eigen::VectorXf c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.size()) {
        c(i) = a(i) + b(i);
    }
}

__global__ void scalar_vector_add(Eigen::VectorXf a, float b, Eigen::VectorXf c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.size()) {
        c(i) = a(i) + b;
    }
}

//this function subtracts two Eigen vectors and stores the result in a third vector
__global__ void vector_subtract(Eigen::VectorXf a, Eigen::VectorXf b, Eigen::VectorXf c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.size()) {
        c(i) = a(i) - b(i);
    }
}

//This function make a square vector operation
__global__ void vector_square(Eigen::VectorXf a, Eigen::VectorXf c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.size()) {
        c(i) = a(i) * a(i);
    }
}


