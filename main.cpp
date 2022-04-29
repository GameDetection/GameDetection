#include <iostream>
#include "Eigen/Core"
#include "inc/mlp_model.h"
#include <chrono>
using namespace Eigen;
using namespace std;

void printLayer(auto layer) {
    for (int layerID = 0; layerID < layer.size(); ++layerID) {
        std::cout << " Layer " << layerID << std::endl;
        for (int neurID = 0; neurID < layer(layerID).size(); ++neurID) {
            std::cout << " neur " << neurID << std::endl;
            std::cout << layer(layerID)(neurID) << std::endl;
        }
    }
}

int main() {
    srand((unsigned int) time(0));

    int tab[3] = {4,3,1};
    MatrixXf input(1,4);
    input.setConstant(10);

    MatrixXf Y(1,1);
    Y(0,0) = 1;

    Mlp_model* model = create_mlp_model(tab,3);

    VectorXf res = predict(model, input.row(0), 1);
    std::cout << "Prediction: " << res << "\n";
    auto start = std::chrono::high_resolution_clock::now();
    train(model, input, Y, 0.0001, 40000, 1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Training time: " << diff.count() << " s\n";

    std::cout << "Input: " << input.row(0) << "\n";
    res = predict(model, input.row(0), 1);
    std::cout << "Prediction: " << res << "\n";
    return 0;
}


