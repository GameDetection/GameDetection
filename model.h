typedef struct matrix {
    int nbc;
    int nbl;
    double** mat;
}  matrix ;

typedef struct layer {
    int m;
    int nbVar;
    matrix input;
    matrix weight;
    matrix cons;
} layer;
typedef struct model {
    matrix* layers;
};

void printMatrix(matrix matrix1);
matrix makeMatrix(int nbC, int nbL, double value);
layer makeLayer(int nbP, int nbVar);
matrix multiplyMatrix(matrix matrix1, matrix matrix2);
matrix addMatrix(matrix matrix1, matrix matrix2);
matrix doZ(layer layer1);