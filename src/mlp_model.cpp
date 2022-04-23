//
// Created by Alexandre Carmone on 02/04/2022.
//
#include "../inc/mlp_model.h"

#include <iostream>

using namespace Eigen;

/**
 * This function create a mlp model
 * @param tab
 * @param len
 * @return
 */
Mlp_model* create_mlp_model(int tab[], int len) {
    // Create a model with the given parameters
    auto* model = new Mlp_model();
    model->inputSize = tab[0];
    model->l.resize(len -1);
    for (int i = 1; i < len; ++i) {
        model->l(i -1 ) = tab[i];
    }
    model->X.resize(len);
    model->B.resize(len);
    model->W.resize(len);
    model->Delta.resize(len);

    for (int i = 0; i < len; ++i) {
        model->Delta(i).resize(tab[i]);
        model->Delta(i).setZero();

        model->X(i).resize(tab[i]);
        model->X(i).setOnes();
        model->W(i).resize(tab[i]);
        model->B(i).resize(tab[i]);
        for (int j = 0; j < tab[i]; ++j) {
            if (i > 0) {
                model->W(i)(j).resize(tab[i - 1]);
                model->W(i)(j).setRandom();

                model->B(i)(j).resize(1);
                model->B(i)(j).setOnes();
            }
        }
    }
    return model;
}

/**
 * This function predict the result of the neural network
 * @param model
 * @param X
 * @param isClassification
 * @return
 */
VectorXf predict(Mlp_model *model, VectorXf X, bool isClassification) {
    model->X(0) = X;
    for (int couchID = 1; couchID < model->X.size(); couchID++) {
        for (int neurID = 0; neurID < model->X(couchID).size(); neurID++) {
            if (model->X.size() - 1 == couchID and isClassification) {
                model->X(couchID)(neurID) = tanh((model->X(couchID - 1).transpose() *
                            model->W(couchID)(neurID) +
                            model->B(couchID)(neurID))
                        .sum());
            }
            else {
                model->X(couchID)(neurID) = (model->X(couchID - 1).transpose() *
                            model->W(couchID)(neurID) +
                            model->B(couchID)(neurID))
                        .sum();
            }
        }
    }
    return model->X(model->X.size() - 1);
}

/**
 * This function train the neural network
 * @param model
 * @param inputs
 * @param Y
 * @param learningRate
 * @param epochs
 * @param isClassification
 */
void train(Mlp_model* model, MatrixXf inputs, MatrixXf Y, float learningRate, int epochs, bool isClassification) {
    int k;
    VectorXf oneInput;
    VectorXf oneOutput;
    for (int i = 0; i < epochs; ++i) {
        k = rand() % inputs.rows();
        oneInput = inputs.row(k);
        oneOutput = Y.row(k);

        predict(model, oneInput, isClassification);

        for (int layerID = model->W.size() - 1; layerID >= 0 ; --layerID) {
            VectorXf One(model->X(layerID).size());
            One.setOnes();
            if (layerID == model->W.size() - 1) {
                model->Delta(layerID) = (One - model->X(layerID).cwiseProduct(model->X(layerID))).cwiseProduct(model->X(layerID) - oneOutput);
            }
            else {
                VectorXf res(model->W(layerID).size());
                res.setZero();
                for (int neurID = 0; neurID < model->Delta(layerID + 1).size(); ++neurID) {
                    res += model->W(layerID + 1)(neurID) * model->Delta(layerID + 1)(neurID);
                }
                model->Delta(layerID) = (One - model->X(layerID).cwiseProduct(model->X(layerID))).cwiseProduct(res);

                for (int neurID = 0; neurID < model->W(layerID).size(); ++neurID) {

                    auto m = learningRate * model->X(layerID + 1) * (model->Delta(layerID)(neurID));
                    std::cout << "m = " << m << std::endl;
                    std::cout << "W = " << model->W(layerID)(neurID) << std::endl;
//                    model->W(layerID)(neurID) = model->W(layerID)(neurID) - learningRate * model->X(layerID + 1).cwiseProduct(model->Delta(layerID + 1)) ;
                }
            }



        }
//        std::cout << "Delta" << std::endl;
//        for (int couchID = 0; couchID < model->Delta.size(); ++couchID) {
//            std::cout << "couche" << std::endl;
//            std::cout << model->Delta(couchID) << std::endl;
//        }

//        for (int layerID = model->W.size() - 1; layerID >= 0 ; --layerID) {
//            std::cout << "Couche = " << layerID << std::endl;
//            VectorXf res(model->W(layerID)(0).size());
//            res.setZero();
//            for (int neurID = 0; neurID < model->Delta(layerID).size(); ++neurID) {
//                std::cout << "neurID = " << neurID << std::endl;
//                res += (model->Delta(layerID)(neurID) * model->W(layerID)(neurID));
//            }
//            std::cout << res << std::endl;
//        }
    }
}
