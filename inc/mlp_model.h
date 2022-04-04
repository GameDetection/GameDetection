//
// Created by Carmo on 02/04/2022.
//
#include <Eigen>
#ifndef GAMEDETECTION_MLP_MODEL_H
#define GAMEDETECTION_MLP_MODEL_H
struct Mlp_model {
    Eigen::VectorXd d;
    int layers;
    Eigen::VectorX<Eigen::MatrixXf> W;
    Eigen::VectorX<Eigen::MatrixXf> B;
    Eigen::VectorX<Eigen::MatrixXf> Xd;
    Eigen::MatrixXf X;
};
Mlp_model create_mlp_model(int tab[], int len);

#endif GAMEDETECTION_MLP_MODEL_H
