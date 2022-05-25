//
// Created by Carmo on 23/05/2022.
//

#ifndef LINEARMODEL_LINEAR_MODEL_H
#define LINEARMODEL_LINEAR_MODEL_H
#include <Eigen>
struct Linear_model {
    Eigen::VectorXf Xk;
    Eigen::VectorXf W;
};

Linear_model create_linear_model(int nbInputs);
Linear_model model_train(Linear_model model, float Xarray[], int Xrows, int Xcols, float Yarray[], int Ylength, int epochs, float learning_rate);
int predict_C(Linear_model model, Eigen::VectorXf X);
float predict(Linear_model model, float Xarray[], int Xlength);
Eigen::MatrixXf getMatrixFromLineMatrix(float table[], int rows, int cols);
Eigen::VectorXf getVectorXfFromLineVector(float table[], int size);

#endif LINEARMODEL_LINEAR_MODEL_H
