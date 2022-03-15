//
// Created by Carmo on 13/03/2022.
//
typedef struct perceptron {
    double w1;
    double w2;
    double x1;
    double x2;
    double b;
} perceptron;

typedef struct couche {
    int len;
    perceptron* couche;
} couche;

double makeZ(perceptron perceptron1) {
    return perceptron1.w1*perceptron1.x1 + perceptron1.w2*perceptron1.x2 + perceptron1.b;
}
perceptron makePerceptron(double x1, double w1, double x2, double w2, double b) {
    perceptron perceptron1;
    perceptron1.x1 = x1;
    perceptron1.x2 = x2;
    perceptron1.w1 = w1;
    perceptron1.w2 = w2;
    perceptron1.b = b;
    return perceptron1;
}
