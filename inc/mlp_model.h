//
// Created by Alexandre Carmone on 02/04/2022.
//
#include <Eigen>
#ifndef GAMEDETECTION_MLP_MODEL_H
#define GAMEDETECTION_MLP_MODEL_H
struct Mlp_model {
    Eigen::VectorXd l;
    int inputSize;
    Eigen::VectorX<Eigen::VectorX<Eigen::VectorXf>> X;
    Eigen::VectorX<Eigen::VectorX<Eigen::VectorXf>> W;
    Eigen::VectorX<Eigen::VectorX<Eigen::VectorXf>> B;
    Eigen::MatrixXf res;

};
Mlp_model create_mlp_model(int tab[], int len);
Eigen::VectorXf predict(Mlp_model model, Eigen::VectorXf X, bool isClassification);
#endif GAMEDETECTION_MLP_MODEL_H