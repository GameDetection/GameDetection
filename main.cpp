#include <iostream>
#include "Eigen/Core"
#include "inc/mlp_model.h"
using namespace Eigen;

int main() {
    int tab[3] = {2, 3,1};
    MatrixXf input(2,2);
    input(0,0) = 2;
    input(0,1) = 5;

    input(1,0) = 7;
    input(1,1) = 8;

    MatrixXf Y(2,1);
    Y(0,0) = 10;
    Y(1,0) = 48;

    Mlp_model* model = create_mlp_model(tab, 3);
//    std::cout << "X" << std::endl;
//    for (int couchID = 0; couchID < model->X.size(); ++couchID) {
//        std::cout << model->X(couchID) << std::endl;
//    }
//    std::cout << "res = " << predict(model, input, 1) << std::endl;
//    std::cout << "Xs" << std::endl;
//    for (int couchID = 0; couchID < model->X.size(); ++couchID) {
//        std::cout << "couche" << std::endl;
//        std::cout << model->X(couchID) << std::endl;
//    }
    train(model, input, Y, 0.002, 1, 1);
    return 0;
}


