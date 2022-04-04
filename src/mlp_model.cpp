//
// Created by Carmo on 02/04/2022.
//
#include "../inc/mlp_model.h"
#include <random>

Mlp_model create_mlp_model(int tab[], int len) {
    // Create a model with the given parameters
    Mlp_model model;
    model.layers = len - 1;

    model.d.resize(len);
    model.X.resize(1,tab[0]);
    model.W.resize(len - 1);
    model.B.resize(len - 1);
    model.Xd.resize(len - 1);

    model.X.setZero();
    for (int i = 0; i < len; ++i) {
        auto nbW = 0;
        if (i + 1 <= len - 1) {
            nbW = tab[i] * tab[i + 1];
            model.W(i).resize(1,nbW);
            model.W(i).setRandom();

            model.Xd(i).resize(1,nbW);
            model.Xd(i).setZero();
        }
        if (i > 0) {
            model.B(i-1).resize(1,tab[i]);
            model.B(i-1).setOnes();

        }
        model.d(i) = tab[i];
    }





    return model;
}