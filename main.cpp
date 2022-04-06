#include <iostream>
#include "Eigen/Core"
#include "inc/mlp_model.h"
using namespace Eigen;
int main() {
    int tab[4] = {2, 2,4,1};
    VectorXf input(2);
    input(0) = 4;
    input(1) = 5;
    Mlp_model model = create_mlp_model(tab, 4);

    auto test = predict(model, input, 1);
    std::cout << "res = " << test << std::endl;
    return 0;
}
