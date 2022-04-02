//
// Created by Carmo on 30/03/2022.
//
#include <Eigen>
#ifndef GAMEDETECTIONPP_MODEL_H
#define GAMEDETECTIONPP_MODEL_H
struct Model {
    Eigen::MatrixXf Xk;
    Eigen::MatrixXf W;
    float b;
};
Model create_linear_model(int nbInput, int b);
Model model_train(Model model, float Xarray[], int Xrows, int Xcols, float Yarray[],int Yrows, int Ycols, int epochs, float learning_rate);
float predict_C(Model model);
float predict(float Xarray[],int Xrows, int Xcols, Model model);
Eigen::MatrixXf getMatrixFromLineMatrix(float table[], int rows, int cols);
#endif GAMEDETECTIONPP_MODEL_H
