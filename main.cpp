#include <iostream>
#include "Eigen/Core"
#include "inc/mlp_model.h"
using namespace Eigen;
int main() {
    int tab[3] = {2, 2,1};
    Mlp_model model = create_mlp_model(tab, 3);
    // print the model
//    std::cout << "model : " << std::endl;
    std::cout << "Weights" << std::endl;
    for (int i = 0; i < model.W.size(); ++i) {
        std::cout << model.W(i) << std::endl;
    }
    std::cout << "Sous couche de X dans le mlp" << std::endl;
    for (int i = 0; i < model.Xd.size(); ++i) {
        std::cout << model.Xd(i) << std::endl;
    }

    std::cout << "Biases" << std::endl;
    for (int i = 0; i < model.B.size(); ++i) {
        std::cout << model.B(i) << std::endl;
    }
    std::cout << "Inputs" << std::endl;
    std::cout << model.X << std::endl;
    return 0;
}
