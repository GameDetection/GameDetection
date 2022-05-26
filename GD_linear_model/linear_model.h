//
// Created by Carmo on 23/05/2022.
//
#include "Eigen"
#ifndef LINEARMODEL_LINEAR_MODEL_H
#define LINEARMODEL_LINEAR_MODEL_H
#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {
struct Linear_model {
    Eigen::VectorXf Xk;
    Eigen::VectorXf W;
};

DLLEXPORT Linear_model *create_linear_model(int nbInputs);
DLLEXPORT Linear_model *create_linear_model_for_regression(float Xarray[], int Xrows, int Xcols, float Yarray[], int Ylength);

DLLEXPORT void train(Linear_model *model, float Xarray[], int Xrows, int Xcols, float Yarray[], int Ylength, int epochs,
           float learning_rate);
DLLEXPORT float predict_C(Linear_model *model, Eigen::VectorXf X);
DLLEXPORT float predict(Linear_model *model, float Xarray[], int Xlength);
DLLEXPORT float predict_for_regression(Linear_model *model, float Xarray[], int Xlength);
}
Eigen::MatrixXf getMatrixXfFromLineMatrix(float table[], int rows, int cols);
Eigen::VectorXf getVectorXfFromLineVector(float table[], int size);

#endif LINEARMODEL_LINEAR_MODEL_H
