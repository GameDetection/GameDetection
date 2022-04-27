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
    int tab[2] = {2,1};
    MatrixXf input(1,2);
    input(0,0) = 10;
    input(0,1) = 10;
//    input(0,2) = 10;
//    input(0,3) = 10;
//    input(0,4) = 10;
//    input(0,5) = 10;
//    input(0,6) = 10;
//    input(0,7) = 10;
//    input(0,8) = 10;
//    input(0,9) = 10;

    MatrixXf Y(1,1);
    Y(0,0) = 10;

    Mlp_model* model = create_mlp_model(tab,2);

//    auto layerW = model->W;
    for (int layerID = 0; layerID < model->W.size(); ++layerID) {
        std::cout << " LayerW " << layerID << std::endl;
        for (int neurID = 0; neurID < model->W(layerID).size(); ++neurID) {
            std::cout << " neurW " << neurID << std::endl;
            std::cout << model->W(layerID)(neurID) << std::endl;
        }
    }
    auto layerX = model->X;
    for (int layerID = 0; layerID < layerX.size(); ++layerID) {
        std::cout << " LayerX " << layerID << std::endl;
        std::cout << layerX(layerID) << std::endl;
    }
    VectorXf res = predict(model, input.row(0), 0);
    std::cout << "Prediction: " << res << "\n";
    auto start = std::chrono::high_resolution_clock::now();
    train(model, input, Y, 0.0001, 100, 0);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Training time: " << diff.count() << " s\n";

    std::cout << "Input: " << input.row(0) << "\n";
    res = predict(model, input.row(0), 0);
    std::cout << "Prediction: " << res << "\n";
    return 0;
}


