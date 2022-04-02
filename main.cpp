#include <iostream>
#include "Eigen/Core"
#include "inc/model.h"
using namespace Eigen;
int main() {

    //initialize input
    float Xarray[6] = {0,0,0,1,1,0};
    float Yarray[3] = {-1,1,1};

    //initialize model

    Model model = create_linear_model(2, -1);
    MatrixXf X = getMatrixFromLineMatrix(Xarray,3,2);
//    std::cout << X << std::endl;
//    MatrixXf Y = getMatrixFromLineMatrix(Yarray,1,3);
//    std::cout << Y << std::endl;
    std::cout << model.W << std::endl;
    model = model_train(model, Xarray,3,2, Yarray,1,3, 12000,0.07);
    std::cout << model.W << std::endl;
    float test;

    for (int i = 0; i < 3; ++i) {
        model.Xk = X.row(i);
        test = predict_C(model);
        std::cout << test << std::endl;
    }
//    float Xarrayk1[2] = {0,0};
//    float Xarrayk2[2] = {0,1};
//    float Xarrayk3[2] = {1,0};
//
//    test = predict(Xarrayk1, 1, 2, model);
//    std::cout << test << std::endl;
//    test = predict(Xarrayk2, 1, 2, model);
//    std::cout << test << std::endl;
//    test = predict(Xarrayk3, 1, 2, model);
//    std::cout << test << std::endl;

    return 0;
}
