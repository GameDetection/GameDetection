#include <iostream>
#include "linear_model.h"

using namespace Eigen;
using namespace std;

int main () {
    Linear_model model = create_linear_model(4);
    float Xarray[4] = {1,2,3,4};
    float Yarray[1] = {1};
    model = model_train(model, Xarray, 1, 4, Yarray, 1, 1, 0.1);
    cout << predict(model, Xarray, 4) << endl;
}
