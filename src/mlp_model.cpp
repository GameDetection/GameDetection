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
    for (int layerID = 1; layerID < model->X.size(); layerID++) {
        for (int neurID = 0; neurID < model->X(layerID).size(); neurID++) {
            // Si nous sommes à la dernière couche et que le model est un classificateur, on applique la tangente hyperbolique
            if (model->X.size() - 1 == layerID and isClassification) {
                model->X(layerID)(neurID) = tanh((model->X(layerID - 1).transpose() *
                          model->W(layerID)(neurID) +
                          model->B(layerID)(neurID))
                        .sum());
            }
            else {
                // pour toutes les autres couches on calcule le résultat de la couche
                model->X(layerID)(neurID) = (model->X(layerID - 1).transpose() *
                         model->W(layerID)(neurID) +
                         model->B(layerID)(neurID))
                        .sum();
            }

        }
    }
    VectorXf a = model->X(model->X.size() - 1);
//    std::cout << " res ="<< a << std::endl;

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
void train(Mlp_model* model, MatrixXf inputs, MatrixXf Y, float learningRate, int epochs, bool isClassification) {
    int k;
    VectorXf Xk;
    VectorXf Yk;
    for (int i = 0; i < epochs; ++i) {
        k = rand() % inputs.rows();
        Xk = inputs.row(k);
        Yk = Y.row(k);

        predict(model, Xk, isClassification);
        //On parcours les couche du model
        for (int layerID = model->W.size() - 1; layerID >= 0 ; --layerID) {
            VectorXf One(model->X(layerID).size());
            One.setOnes();
            // si on est dans la derniere couche
             // Ici on calcule de le Delta de la dernière couche
             //TODO: en fait c'est sans doute ici que ça plante

            if (layerID == model->W.size() - 1) {
//                auto m = One - model->X(layerID).cwiseProduct(model->X(layerID));
//                std::cout << "m = " << m << std::endl;
//                auto n = model->X(layerID) - Yk;
//                std::cout << "test X^2 = " << model->X(layerID).cwiseProduct(model->X(layerID)) << std::endl;
//                auto test = One - (model->X(layerID).cwiseProduct(model->X(layerID)));//
//                auto test2 = model->X(layerID) - Yk;
//                std::cout << "test 1 - x^2 = " << test << std::endl;
//                std::cout << "test x - y = " << test2 << std::endl;
                model->Delta(layerID) = (One -
                        model->X(layerID).cwiseProduct(model->X(layerID))).cwiseProduct(model->X(layerID) - Yk);
//                std::cout << "Delta test = " << (One -
//                                                 model->X(layerID).cwiseProduct(model->X(layerID))).cwiseProduct(model->X(layerID) - Yk) << std::endl;
                std::cout << "Delta L = " << model->Delta(layerID) << std::endl;
            }
            // pour les autres couches
            else {
                VectorXf res(model->W(layerID).size());
                res.setZero();
                // pour chaque neuronne dans la couche
                // cette partie fonctionne très mal

                 // ensuite on calcule le delta pour toutes les autres couches
                 // TODO: regarder cette partie pour que le delta soit correct
                for (int neurID = 0; neurID < model->Delta(layerID + 1).size(); ++neurID) {
                    res += model->W(layerID + 1)(neurID) * model->Delta(layerID + 1)(neurID);
                }
                model->Delta(layerID) = (One - model->X(layerID).cwiseProduct(model->X(layerID))).cwiseProduct(res);
//                std::cout << "Delta= " << model->Delta(layerID) << std::endl;

            }
            if (layerID > 0) {
                //On fait la mise à jour des poids
                for (int neurID = 0; neurID < model->W(layerID).size(); ++neurID) {
                    std::cout << "W avant = " << model->W(layerID)(neurID) << std::endl;
                    model->W(layerID)(neurID) -= learningRate * model->X(layerID - 1) * model->Delta(layerID)(neurID);
                    std::cout << "W apres = " << model->W(layerID)(neurID) << std::endl;
                }
            }
        }
    }
}
