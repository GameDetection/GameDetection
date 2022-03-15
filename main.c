#include <stdio.h>
#include <stdlib.h>
#include "inc/model.h"

int main() {

//    matrix matrix1 = makeMatrix(2,2,5);
//
//    matrix1.mat[0][0] = 4;
//    matrix1.mat[0][1] = 5;
//
//    matrix1.mat[1][0] = 2;
//    matrix1.mat[1][1] = 8;
//    matrix1.mat = {
//            {2.0,3.0},
//            {}
//    };
//    printMatrix(matrix1);
//    printf("\n");
//    matrix matrix2 = makeMatrix(2,2,3);
//    printMatrix(matrix2);
//    printf("\n");
//
//    matrix matrix3 = multiplyMatrix(matrix1, matrix2);
//    printMatrix(addMatrix(matrix1, matrix2));
//    printf("\n");
//    printMatrix(matrix3);

    layer layer1 = makeLayer(3,2);
    printMatrix(layer1.cons);
    printf("\n");
    printMatrix(layer1.weight);
    printf("\n");
    printMatrix(layer1.input);
    printf("\n");
    printMatrix(doZ(layer1));

}
