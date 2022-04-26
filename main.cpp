#include <iostream>
#include "Eigen/Core"
#include "inc/mlp_model.h"
#include <chrono>
using namespace Eigen;

int main() {
    int tab[4] = {10, 60, 150,1};
    MatrixXf input(1,10);
    input(0,0) = 2;
    input(0,1) = 5;
    input(0,2) = 6;
    input(0,3) = 7;
    input(0,4) = 8;
    input(0,5) = 9;
    input(0,6) = 10;
    input(0,7) = 11;
    input(0,8) = 12;
    input(0,9) = 13;

//    input(1,0) = 7;
//    input(1,1) = 8;

    MatrixXf Y(1,1);
    Y(0,0) = 10;

    Mlp_model* model = create_mlp_model(tab, 4);

    auto start = std::chrono::high_resolution_clock::now();
    train(model, input, Y, 0.002, 50000, 1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Training time: " << diff.count() << " s\n";


    return 0;
}


