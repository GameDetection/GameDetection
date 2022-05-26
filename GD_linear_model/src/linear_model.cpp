//
// Created by Carmo on 23/05/2022.
//
#include <Eigen>
#include <random>
#include <iostream>
#include <utility>
#include "../linear_model.h"
using namespace Eigen;
using namespace std;


/**
 * This function is used to predict de result of the model in c++
 * @param model
 * @return
 */
float predict_C(Linear_model* model, VectorXf X) {
    model->Xk = std::move(X);
    if ((model->W.transpose() * model->Xk)(0) >= 0) {
        return 1.0;
    }
    else {
        return -1.0;
    }
}

/**
 * This function is used to predict de result of the model
 * @param model
 * @param Xarray
 * @param Xlength
 * @return
 */
float predict(Linear_model* model, float Xarray[], int Xlength) {
    model->Xk = getVectorXfFromLineVector(Xarray, Xlength);
    if ((model->W.transpose() * model->Xk)(0) >= 0) {
        return 1;
    }
    else {
        return -1;
    }
}
float predict_for_regression(Linear_model* model, float Xarray[], int Xlength) {
    model->Xk = getVectorXfFromLineVector(Xarray, Xlength);
    return (model->W.transpose() * model->Xk)(0);
}


/**
 * This function is used to convert a float table to a Eigen Vector
 * @param table
 * @param size
 * @return
 */
VectorXf getVectorXfFromLineVector(float table[], int size) {
    VectorXf res;
    res.resize(size + 1);
    res = VectorXf::Map(table, size + 1);
    res(size) = 1;
    return res;
}
/**
 * This function is used to convert a line matrix (table of float) to a Eigen matrix
 * @param table
 * @param rows
 * @param cols
 * @return
 */
MatrixXf getMatrixXfFromLineMatrix(float table[], int rows, int cols) {
    MatrixXf res;
    res.resize(rows, cols + 1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i, j) = table[i * cols + j];
        }
        res(i, cols) = 1;
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
void train(Linear_model* model, float Xarray[], int Xrows, int Xcols, float Yarray[], int Ylength, int epochs, float learning_rate) {

    auto X = getMatrixXfFromLineMatrix(Xarray, Xrows, Xcols);
    auto Y = getVectorXfFromLineVector(Yarray, Ylength);
    int k = 0;
    int Yk = 0;
    float GXk = 0;
    for (int j = 0; j < epochs; ++j) {

        k = rand() % X.rows();

        VectorXf Xk = X.row(k);


        GXk = predict_C(model, Xk);
//        cout << "GXk: " << GXk << endl;
//        cout << " Calc: " << endl << learning_rate * (Y(k) - GXk) << endl;
        model->W = model->W + learning_rate * (Y(k) - GXk) * Xk;
    }
}
/**
 * This function creates a model with random weights
 * @param nbInput
 * @return model
 */
Linear_model* create_linear_model(int nbInputs) {
    Linear_model* model = new Linear_model;

    model->Xk.resize(nbInputs + 1);
    model->W.resize(nbInputs + 1);

    model->Xk.setOnes();
    model->W.setRandom();

    return model;
}


Linear_model* create_linear_model_for_regression(float Xarray[], int Xrows, int Xcols, float Yarray[], int Ylength) {
    Linear_model *model = new Linear_model;


    VectorXf Y;
    Y.resize(Ylength);
    Y = VectorXf::Map(Yarray, Ylength);
    auto X = getMatrixXfFromLineMatrix(Xarray, Xrows, Xcols);

    VectorXf W = ((X.transpose() * X).completeOrthogonalDecomposition().pseudoInverse() *X.transpose()) * Y;

    model->W = W;
    return model;
}

