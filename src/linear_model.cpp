//
// Created by Carmo on 30/03/2022.
//
#include <Eigen>
#include <random>
#include <iostream>
#include "../inc/linear_model.h"
using namespace Eigen;

/**
 * This function is used to predict de result of the model in c++
 * @param model
 * @return
 */
float predict_C(Linear_model model) {
    float total;
    MatrixXf res = model.Xk * model.W.transpose();
    total = res(0,0) + model.b;
    if (total >= 0) {
        return 1;
    }
    else {
        return -1;
    }
}

/**
 * This function is used to predict de result of the model in a lib
 * @param Xarray
 * @param Xrows
 * @param Xcols
 * @param model
 * @return
 */
float predict(float Xarray[], int Xrows, int Xcols, Linear_model model) {
    model.b;
    model.W;
    MatrixXf Xk = getMatrixFromLineMatrix(Xarray, Xrows, Xcols);
    float total;
    MatrixXf res = Xk * model.W.transpose();
    total = res(0,0) + model.b;
    if (total >= 0) {
        return 1;
    }
    else {
        return -1;
    }
}

/**
 * This function convert a line matrix to a Matrix
 * @param table
 * @param rows
 * @param cols
 * @return
 */
MatrixXf getMatrixFromLineMatrix(float table[], int rows, int cols) {
    MatrixXf res;
    res.resize(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i,j) = table[i*cols + j];
        }
    }
    return res;
}

/**
 * This function train the model with the given data
 * @param model
 * @param Xarray
 * @param Xrows
 * @param Xcols
 * @param Yarray
 * @param Yrows
 * @param Ycols
 * @param epochs
 * @param learning_rate
 * @return
 */
Linear_model model_train(Linear_model model, float Xarray[], int Xrows, int Xcols, float Yarray[], int Yrows, int Ycols, int epochs, float learning_rate) {

    MatrixXf X = getMatrixFromLineMatrix(Xarray, Xrows, Xcols);
    MatrixXf Y = getMatrixFromLineMatrix(Yarray, Yrows, Ycols);
    int k = 0;
    for (int j = 0; j < epochs; ++j) {
        k = rand() % X.rows();

        float Yk = Y(0,k);
        model.Xk = X.row(k);


//        float Wp = predict(model);

        float total;
        float Wp;
        MatrixXf res = X.row(k) * model.W.transpose();
        total = res(0,0) + model.b;
        if (total >= 0) {
            Wp = 1;
        }
        else {
            Wp = -1;
        }


        MatrixXf matrixTemp;
        matrixTemp.resize(1,1);
        matrixTemp.setConstant(learning_rate * (Yk - Wp));
        MatrixXf matrixTemp2;
        matrixTemp2.resize(1,2);
        matrixTemp2.setOnes();
        model.W = model.W + (matrixTemp * (X.row(k) + matrixTemp2));
    }
    return model;
}
/**
 * This function creates a model with random weights
 * @param nbInput
 * @param b
 * @return model
 */
Linear_model create_linear_model(int nbInput, int b) {
    Linear_model model;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-0.99, 0.99);
    model.Xk.resize(1,nbInput);
    model.W.resize(1,nbInput);
    model.b = b;
    for (int i = 0; i < nbInput; ++i) {
        model.Xk(0,i) = 0;
        model.W(0,i) = dist(gen);
    }
    return model;
}

