//
// Created by Carmo on 13/03/2022.
//
#include <stdio.h>
#include <stdlib.h>
#include "inc/matrix.h"

/**
 * @param matrix1
 * affiche la matrice
 */
void printMatrix(matrix matrix1) {
    double **mat = matrix1.mat;
    for (int i = 0; i < matrix1.nbl; i++) {
        for (int j = 0; j < matrix1.nbc; ++j) {
            printf("%lf ", mat[i][j]);
        }
        printf("\n");
    }
}
/**
 * @param nbC nombre de colonnes
 * @param nbL nombre de lignes
 * @param value valeur à mettre dans la matrice
 * @return une matrice de taille nbL*nbC avec toutes les valeurs égales à value
 */
matrix makeMatrix(int nbC, int nbL, double value) {
    double **matrix1 = malloc(nbL * sizeof(double *));

    matrix matrix2;
    matrix2.nbc = nbC;
    matrix2.nbl = nbL;
    matrix2.mat = matrix1;

    for (int i = 0; i < nbL; i++) {
        double *tab = malloc(nbC * sizeof(double));
        for (int j = 0; j < nbC; j++) {
            tab[j] = value;
        }
        matrix1[i] = tab;
    }
    return matrix2;
}

/**
 * @param matrix1
 * @param matrix2
 * @return retourne la matrice issus de la multiplication de deux matrices
 */
matrix multiplyMatrix(matrix matrix1, matrix matrix2) {

    matrix newMatrix = makeMatrix(matrix2.nbc,matrix1.nbl,0);
    if(matrix1.nbc == matrix2.nbl) {
        double res = 0;
        for (int i = 0; i < matrix1.nbl; i++) {
            for (int j = 0; j < matrix1.nbc; ++j) {
                for (int k = 0; k < matrix1.nbc; k++) {
                    newMatrix.mat[i][j] += matrix1.mat[i][k] * matrix2.mat[k][j];
                }
            }
        }
    } else {
        printf("dimention incorecte\n");
    }
    return newMatrix;
}
/**
 * @return retourne la matrice issus de l'addition de deux matrices
 * @param matrix1
 * @param matrix2
 */
matrix addMatrix(matrix matrix1, matrix matrix2) {
    matrix newMatrix = makeMatrix(matrix2.nbc,matrix1.nbl,0);
//    printf("%d %d, %d %d", matrix2.nbc, matrix1.nbc, matrix1.nbl)
    if (matrix2.nbc == matrix1.nbc && matrix1.nbl == matrix2.nbl) {
        for (int i = 0; i < matrix1.nbl; ++i) {
            for (int j = 0; j < matrix2.nbc; ++j) {
                newMatrix.mat[i][j] = matrix1.mat[i][j] + matrix2.mat[i][j];
            }
        }
    } else{
        printf("Dimentions incorecte\n");
    }
    return newMatrix;
}



