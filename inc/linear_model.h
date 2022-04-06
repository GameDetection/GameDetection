//
// Created by Alexandre Carmone on 30/03/2022.
//
#include <Eigen>
#ifndef GAMEDETECTIONPP_MODEL_H
#define GAMEDETECTIONPP_MODEL_H
struct Linear_model {
    Eigen::MatrixXf Xk;
    Eigen::MatrixXf W;
    float b;
};
Linear_model create_linear_model(int nbInput, int b);
Linear_model model_train(Linear_model model, float Xarray[], int Xrows, int Xcols, float Yarray[], int Yrows, int Ycols, int epochs, float learning_rate);
float predict_C(Linear_model model);
float predict(float Xarray[], int Xrows, int Xcols, Linear_model model);
Eigen::MatrixXf getMatrixFromLineMatrix(float table[], int rows, int cols);
#endif GAMEDETECTIONPP_MODEL_H
