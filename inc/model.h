//
// Created by Carmo on 15/03/2022.
//

#ifndef GAMEDETECTION_MODEL_H
#define GAMEDETECTION_MODEL_H

#include "matrix.h"

/**
 * structure de type layer contenant les matrices d'entrer, de poids et de constantes
 * Ainsi que le nombre de neurones dans la couche et de variable d'entrainement
 */
typedef struct layer {
    int m;
    int nbVar;
    matrix input;
    matrix weight;
    matrix cons;
} layer;
/**
 * structure de type model contenant les couches du reseau de neurone
 */
typedef struct model {
    int nbLayer;
    matrix* layers;
};

// déclaration des prototypes de fonctions

layer makeLayer(int nbP, int nbVar);

matrix doZ(layer layer1);
matrix doA(matrix Z);
#endif GAMEDETECTION_MODEL_H
