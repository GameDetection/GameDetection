//
// Created by Alexandre Carmone on 02/04/2022.
//
#include "../inc/mlp_model.h"

#include <iostream>

using namespace Eigen;
Mlp_model create_mlp_model(int tab[], int len) {
    // Create a model with the given parameters
    Mlp_model model;
    model.inputSize = tab[0];
    model.l.resize(len -1);
    for (int i = 1; i < len; ++i) {
        model.l(i -1 ) = tab[i];
    }
    model.X.resize(len -1 );
    model.B.resize(len -1 );
    model.W.resize(len -1 );
//    model.W.resize(len);
//    model.B.resize( len);


    for (int i = 1; i < len; ++i) {

        model.X(i - 1).resize(tab[i]);
        model.W(i - 1).resize(tab[i]);
        model.B(i - 1).resize(tab[i]);
        for (int j = 0; j < tab[i]; ++j) {

            model.X(i - 1)(j).resize(tab[i - 1]);
            model.X(i - 1)(j).setOnes();

            model.W(i - 1)(j).resize(tab[i - 1]);
            model.W(i - 1)(j).setRandom();

            model.B(i - 1)(j).resize(1);
            model.B(i - 1)(j).setOnes();
        }

    }

    return model;
}


VectorXf predict(Mlp_model model, VectorXf X, bool isClassification) {
    VectorXf result_;

    for (int i = 0; i < model.X(0).size(); ++i) {
        model.X(0)(i) = X;
    }
    for (int couchID = 0; couchID < model.X.size(); couchID++) {

        VectorXf Z(model.X(couchID).size());
        for (int neurID = 0; neurID < model.X(couchID).size(); neurID++) {
            Z(neurID) = (model.X(couchID)(neurID).transpose() * model.W(couchID)(neurID) + model.B(couchID)(neurID)).sum();

        }
        if (couchID < model.X.size() -1) {
            for (int nextNeur = 0; nextNeur < model.X(couchID + 1).size(); nextNeur++) {
                model.X(couchID + 1)(nextNeur) = Z;
            }
        }
        else {
            if (isClassification) {
                for (int i = 0; i < Z.size(); ++i) {
                 Z(i) = tanh(Z(i));
                }
            }
            return Z;
        }
    }

    return result_;
}
