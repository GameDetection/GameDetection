//
// Created by Carmo on 03/05/2022.
//
#include <iostream>
#include <chrono>
#include "mlp_model.cuh"

using namespace std;

int main() {
//    srand((unsigned int) time(0));
    int len = 3;
    int *tab = new int[3]{2, 2, 1};

    try {
        Mlp_model_cuda *model = create_mlp_model(tab, len);
//        auto length = model->XLayer[model->lLenght - 1].length;
//        auto lastX = model->XLayer[model->lLenght - 1].vector;

        int nbFeatures = 2;
        int nbSamples = 10;
        //create a random linear matrix
        auto *X = new float[nbSamples * nbFeatures];
        for (int i = 0; i < nbSamples * nbFeatures; i++) {
            X[i] = 5;
            cout << X[i] << " ";
        }
        cout << endl;

        int nbFeaturesOutputs = 1;
        int nbSamplesOutputs = 10;
        auto *Y = new float[nbSamplesOutputs * nbFeaturesOutputs];
        for (int i = 0; i < nbSamplesOutputs * nbFeaturesOutputs; i++) {
            Y[i] = 1;
        }
//        Xlayer x1;
//        x1.vector = &X[0];
//        x1.length = 10;
//        predict(model, x1, 1);

//        cout << "X test" << endl;
//        for (int layerID = 0; layerID < 4; layerID++) {
//            cout << "layer" << endl;
//            for(int elementID = 0; elementID < model->XLayer[layerID].length; elementID++) {
//                cout << model->XLayer[layerID].vector[elementID] << " ";
//            }
//            cout << endl;
//        }

//        cuda_train(model, X, nbSamples, nbFeatures, Y, nbSamplesOutputs, nbFeaturesOutputs, 1000, 0.004, 1);
        host_train(model, X, nbSamples, nbFeatures, Y, nbSamplesOutputs, nbFeaturesOutputs, 2, 0.004, 1);

//        delete model;
    }
    catch (const char *message) {
        cout << message << endl;
    }
    return 0;
}

