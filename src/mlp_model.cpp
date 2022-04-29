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
Mlp_model *create_mlp_model(int tab[], int len) {
    // Create a model with the given parameters
    auto *model = new Mlp_model();
    model->inputSize = tab[0];
    model->l.resize(len - 1);
    for (int i = 1; i < len; ++i) {
        model->l(i - 1) = tab[i];
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
    for (int layerID = 1; layerID < model->X.size(); layerID++) {
        for (int neurID = 0; neurID < model->X(layerID).size(); neurID++) {

            if (model->X.size() - 1 == layerID) {
                if (isClassification) {
                    model->X(layerID)(neurID) = tanh((model->X(layerID - 1).transpose() *
                                                      model->W(layerID)(neurID) +
                                                      model->B(layerID)(neurID))
                                                             .sum());
                } else {
                    model->X(layerID)(neurID) = ((model->X(layerID - 1).transpose() *
                                                  model->W(layerID)(neurID) +
                                                  model->B(layerID)(neurID))
                            .sum());
                }
            } else {
                model->X(layerID)(neurID) = tanh((model->X(layerID - 1).transpose() *
                                                  model->W(layerID)(neurID) +
                                                  model->B(layerID)(neurID))
                                                         .sum());
            }
        }
    }
    VectorXf a = model->X(model->X.size() - 1);

    return a;
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
void train(Mlp_model *model, MatrixXf inputs, MatrixXf Y, float learningRate, int epochs, bool isClassification) {
    int k;
    VectorXf Xk;
    VectorXf Yk;
    for (int i = 0; i < epochs; ++i) {
//        system("cls");
//        std::cout << "Epoch: "<< i << std::endl;
        k = rand() % inputs.rows();
        Xk = inputs.row(k);
        Yk = Y.row(k);


        predict(model, Xk, isClassification);
        //On parcours les couche du model
        for (int layerID = model->W.size() - 1; layerID >= 0; --layerID) {

            VectorXf One(model->X(layerID).size());
            One.setOnes();
            // Ici on calcule de le Delta de la dernière couche
            if (layerID == model->W.size() - 1) {
                if (isClassification) {
                    //on fait le calcul de la dernière couche en fonction de la classification
                    model->Delta(layerID) = (One - model->X(layerID).cwiseProduct(model->X(layerID))).cwiseProduct(
                            model->X(layerID) - Yk);
                } else {
                    model->Delta(layerID) = model->X(layerID) - Yk;
                }
            }
                // pour les autres couches
            else {
                VectorXf res(model->W(layerID).size());
                res.setZero();
                // pour chaque neuronne dans la couche
                for (int neurID = 0; neurID < model->Delta(layerID + 1).size(); ++neurID) {
                    res += model->W(layerID + 1)(neurID) * model->Delta(layerID + 1)(neurID);
                }
                //on calcule le delta de la couche
                model->Delta(layerID) = (One - model->X(layerID).cwiseProduct(model->X(layerID))).cwiseProduct(res);
            }
            if (layerID > 0) {
                //On fait la mise à jour des poids
                for (int neurID = 0; neurID < model->W(layerID).size(); ++neurID) {
                    model->W(layerID)(neurID) -= learningRate * model->X(layerID - 1) * model->Delta(layerID)(neurID);
                }
            }
        }
    }
}
