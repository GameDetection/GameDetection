//
// Created by Carmo on 15/03/2022.
//

#include "inc/model.h"
#include <math.h>
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

matrix doA(matrix Z) {
    matrix A = makeMatrix(Z.nbc,Z.nbl,1);
    for (int i = 0; i < Z.nbl; i++) {
        for (int j = 0; j < Z.nbc; j++){
            A.mat[i][j] = 1 / (1 + exp(-(Z.mat[i][j])));
        }
    }
    return A;

}
//Pas concluant
double logLoss(matrix A, matrix y){
    double lL = 0;
    matrix mat1 = makeMatrix(1,1,1);
    matrix part1 = multiplyMatrix(changeSigneMatrix(y),lnMatrix(A));
    matrix part2 = multiplyMatrix(lessMatrix(mat1,y), lnMatrix(lessMatrix(mat1,A)));
    lL = (1 / y.nbl * sumMatrix(lessMatrix(part1, part2)));

    return lL;
}
matrix* gradients(matrix A, matrix X, y) {
    matrix dW = ;
    matrix db = ;
}