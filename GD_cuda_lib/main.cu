//
// Created by Carmo on 03/05/2022.
//
#include <iostream>
#include "mlp_model.cuh"
int main() {
    srand((unsigned int) time(0));
    //create model
    int nlp[3] = {2,  1};
    Mlp_model *model = create_mlp_model(nlp, 2);
    std::cout << &model << std::endl;
    //create data input
    float *input = new float[2];
    input[0] = 1;
    input[1] = 1;
    float *output = new float[1];
    output[0] = 0;
    // test model prediction
    predict_for_dll(model, input, 2, 1);

    return 0;
}
