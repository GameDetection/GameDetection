#include <iostream>
#include "linear_model.h"
using namespace std;
int main() {
    //create a dataSet for regression
    float X[8] = {1, 1, 2, 2, 3,3,4,4};
    float Xx[4][2] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    float Y[4] = {4, 3, 2, 1};
    Linear_model *model = create_linear_model_for_regression(X, 4, 2, Y, 4);
    for (int i = 0; i < 4; ++i) {
        cout << "The predicted value is: " << predict_for_regression(model, Xx[i], 2) << endl;
    }






    return 0;
}
