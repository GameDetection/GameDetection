//
// Created by Carmo on 23/05/2022.
//
#include <Eigen>
#include <random>
#include <iostream>
#include "../../GD_cpp_linear_model/linear_model.h"
using namespace Eigen;
using namespace std;

VectorXf getVectorXfFromLineVector(float table[], int size) {
    VectorXf res;
    res.resize(size);
    res = VectorXf::Map(table, size);
    return res;
}

/**
 * This function is used to predict de result of the model in c++
 * @param model
 * @return
 */
int predict_C(Linear_model model, VectorXf X) {
    model.Xk = X;
    auto res = (model.Xk * model.W.transpose())(0) + 1;
    if (res >= 0) {
        return 1;
    }
    else {
        return -1;
    }
}

/**
 * This function is used to predict de result of the model
 * @param model
 * @param Xarray
 * @param Xlength
 * @return
 */
float predict(Linear_model model, float Xarray[], int Xlength) {
    model.Xk = getVectorXfFromLineVector(Xarray, Xlength);
    auto res = (model.Xk * model.W.transpose())(0) + 1;
    if (res >= 0) {
        return 1;
    }
    else {
        return -1;
    }
}

/**
 *
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
 *
 * @param model
 * @param Xarray
 * @param Xrows
 * @param Xcols
 * @param Yarray
 * @param Ylength
 * @param epochs
 * @param learning_rate
 * @return
 */
Linear_model model_train(Linear_model model, float Xarray[], int Xrows, int Xcols, float Yarray[], int Ylength, int epochs, float learning_rate) {

    auto X = getMatrixFromLineMatrix(Xarray, Xrows, Xcols);
    cout << "X : " << X << endl;
    auto Y = getVectorXfFromLineVector(Yarray, Ylength);
    int k = 0;
    for (int j = 0; j < epochs; ++j) {

        k = rand() % X.rows();
        auto Xk = X.row(k);
        float Yk = Y(k);
        float GXk = predict_C(model, Xk);
        auto a1 = learning_rate * (Yk - GXk);
        cout << "a1 : " << a1 << endl;
        cout << "Xk : " << Xk << endl;
        cout << "W : " << model.W << endl;
        model.W = model.W + Xk;
    }
    return model;
}
/**
 * This function creates a model with random weights
 * @param nbInput
 * @return model
 */
Linear_model create_linear_model(int nbInputs) {
    Linear_model model;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-0.99, 0.99);
    model.Xk.resize(nbInputs);
    model.W.resize(nbInputs);
    for (int i = 0; i < nbInputs; ++i) {
        model.Xk(i) = 0;
        model.W(i) = dist(gen);
    }
    return model;
}


