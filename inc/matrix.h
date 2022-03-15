//
// Created by Carmo on 15/03/2022.
//

#ifndef GAMEDETECTION_MATRIX_H
#define GAMEDETECTION_MATRIX_H
/**
 * structure de type matrix contenant ses dimmentions et ses valeurs
 */
typedef struct matrix {
    int nbc;
    int nbl;
    double** mat;
}  matrix ;

void printMatrix(matrix matrix1);
matrix makeMatrix(int nbC, int nbL, double value);

matrix multiplyMatrix(matrix matrix1, matrix matrix2);
matrix addMatrix(matrix matrix1, matrix matrix2);
#endif //GAMEDETECTION_MATRIX_H
