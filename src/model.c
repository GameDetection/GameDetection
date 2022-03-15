//
// Created by Carmo on 15/03/2022.
//

#include "inc/model.h"

/**
 * @param m nombre de neurones/perceptron
 * @param nbVar nombe de variable de la couche d'entrée
 * @return retourne la couche de neurones
 */
layer makeLayer(int m, int nbVar) {

    matrix input = makeMatrix(nbVar,m,1);

    matrix weight = makeMatrix(1,nbVar,1);

    matrix cons = makeMatrix(1,m,1);

    layer layer1;
    layer1.input = input;
    layer1.weight = weight;
    layer1.cons = cons;
    return layer1;
}
/**
 *
 * @param layer1
 * @return retourne le resulat Z sous forme de matrice de la couche de neurones
 */
matrix doZ(layer layer1) {
    return addMatrix(layer1.cons,multiplyMatrix(layer1.input, layer1.weight));
}