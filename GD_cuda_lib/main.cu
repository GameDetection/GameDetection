//
// Created by Carmo on 03/05/2022.
//
#include <iostream>
#include <chrono>
#include "mlp_model.cuh"
#include "vector_operator.cuh"
using namespace Eigen;
using namespace std;




//make a cuda kernel witch multiply two matrix
__global__ void vector_multiply(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] * B[i];
    }
}


int main() {
    int n = 10000;


    std::chrono::duration<double> diff;
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();

    //declare a Eigen vector of size 10
    VectorXf v1(n);
    v1.setRandom();

    VectorXf v2(n);
    v2.setConstant(2);
//    cout << "v1: " << v1.transpose() << endl;
//    cout << "v2: " << v2.transpose() << endl;
    VectorXf v3(n);
    v3.setZero();



    start = std::chrono::high_resolution_clock::now();
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];



    float *p_a = new float[n];
    float *p_b = new float[n];
    float *p_c = new float[n];
    size_t size = sizeof(float) * n;
//    cout << v3.transpose() << endl;


    cudaMalloc(&p_a, size);
    cudaMalloc(&p_b, size);
    cudaMalloc(&p_c, size);

    cudaMemcpy(p_a, v1.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(p_b, v2.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(p_c, v3.data(), size, cudaMemcpyHostToDevice);

    vector_multiply<<<n,n>>>(p_a, p_b, p_c, n);

    cudaMemcpy(v3.data(), p_c, size, cudaMemcpyDeviceToHost);
    v3.sum();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    cout << "time : " << diff.count() << " s" << endl;
//    cout << v3.transpose() << endl;


    start = std::chrono::high_resolution_clock::now();
    v3 = v1 * v2.transpose();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    cout << "time : " << diff.count() << " s" << endl;


//    vector_multiply<<<1, 1>>>(, v.data(), v.data(), v.size());

    return 0;
}
