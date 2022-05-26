//
// Created by Carmo on 24/05/2022.
//
#include <random>
#include <iostream>
#include "../mlp_model.cuh"
#include <chrono>

using namespace std;

void hostCalculateDeltaLayerForClassification(float *Delta, int nbNeur, float *X, float *Y) {
    for (int i = 0; i < nbNeur; i++) {
        Delta[i] = (1 - X[i] * X[i]) * (X[i] - Y[i]);
        cout << "X[i] : " << X[i] << endl;
        cout << "Calc1 : " << (1 - X[i] * X[i]) << endl;
        cout << "Delta : " << Delta[i] << endl;
    }
    cout << endl;
}

void hostCalculateDeltaLayer(float *Delta, int nbNeur, float *X, float *Y) {
    for (int i = 0; i < nbNeur; i++) {
        Delta[i] = X[i] - Y[i];
    }
}

void hostCalculateWeightAndDeltaForDelta(float *Delta, float *resLayer, float *W, int nbNeur, int neurSize) {
    for (int i = 0; i < nbNeur * neurSize; i++) {
        cout << "W" << W[i] << endl;
        cout << "Delta: " << Delta[i % neurSize] << endl;
        resLayer[i] = Delta[i % neurSize] * W[i];
    }
}

void hostCalculateSumForDelta(float *resLayer, int nbNeur, int neurSize, float *Delta) {
    for (int i = 0; i < nbNeur; i++) {
        Delta[i] = 0;
        for (int j = 0; j < neurSize + (i * neurSize); j++) {
            Delta[i] += resLayer[j];
        }
    }
}

void hostCalculateDelta(float *Delta, float *X, int nbNeur) {
    for (int i = 0; i < nbNeur; i++) {
        Delta[i] = (1 - X[i] * X[i]) * Delta[i];
    }
}

void hostUpdateWeights(float *W, float *Delta, float *X, int nbNeur, int neurSize, float learningRate) {
    for (int i = 0; i < nbNeur * neurSize; i++) {
        W[i] -= Delta[i % nbNeur] * X[i] * learningRate;
    }
}

void hostSumLayer(float *X, float *Wres, int neurSize, int nbNeur) {
    for (int i = 0; i < nbNeur; i++) {
        X[i] = 0;
        for (int j = 0; j < neurSize + (i * neurSize); j++) {
            X[i] += Wres[j];
        }
        cout << "X : " << X[i] << endl;
    }
}

void hostSumTanLayer(float *X, float *Wres, int neurSize, int nbNeur) {
    for (int i = 0; i < nbNeur; i++) {
        X[i] = 0;
        for (int j = 0; j < neurSize + (i * neurSize); j++) {
            X[i] += Wres[j];
        }
        X[i] = tanh(X[i]);
        cout << "X : " << X[i] << endl;
    }
}

void hostCalculateLayer(float *W, float *X, float *Wres, int neurSize, int nbNeur) {
    for (int i = 0; i < neurSize * nbNeur; i++) {
        Wres[i] = W[i] * X[i % neurSize];
        cout << "Wres[" << i << "] : " << Wres[i] << endl;
    }
}

void host_predict(Mlp_model_cuda *model, float *X, int32_t num_features, bool isClassification) {


    model->XLayer[0].vector = X;
    model->XLayer[0].length = num_features;
    int neurSize = 0;
    int nbNeur = 0;

    for (int layerID = 1; layerID < model->lLenght; layerID++) {
        nbNeur = model->WLayer[layerID].nbNeur;
        neurSize = model->WLayer[layerID].neurSize;

        hostCalculateLayer
                (model->WLayer[layerID].vector,
                 model->XLayer[layerID - 1].vector,
                 model->WResLayer[layerID].vector,
                 model->WLayer[layerID].neurSize,
                 model->WLayer[layerID].nbNeur
                );
        if (layerID == (model->lLenght - 1) && !isClassification) {
            hostSumLayer(
                    model->XLayer[layerID].vector,
                    model->WResLayer[layerID].vector,
                    neurSize,
                    nbNeur
            );
        } else {
            hostSumTanLayer(
                    model->XLayer[layerID].vector,
                    model->WResLayer[layerID].vector,
                    neurSize,
                    nbNeur
            );
        }
    }
}

void host_train(Mlp_model_cuda *model, float *all_samples_inputs, int32_t num_samples, int32_t num_features,
                float *all_samples_expected_outputs, int32_t num_samples_outputs, int32_t num_features_outputs,
                int32_t epochs, float learningRate, bool isClassification) {
    auto X = getMatrixXfFromLineMatrix(all_samples_inputs, num_samples, num_features);
    auto Y = getMatrixXfFromLineMatrix(all_samples_expected_outputs, num_samples_outputs, num_features_outputs);
    cout << "Data convert to matrix" << endl;
    auto **d_inputs = new float *[num_samples];
    auto **d_outputs = new float *[num_samples];

    float *Xk;
    float *Yk;
    int k = 0;
//    for (int sample = 0; sample < num_samples; sample++) {
//        cudaMalloc(&(d_inputs[sample]), sizeof(float) * num_features);
//        cudaMemcpy(d_inputs[sample], X[sample], sizeof(float) * num_features, cudaMemcpyHostToDevice);
//
//        cudaMalloc(&(d_outputs[sample]), sizeof(float) * num_features_outputs);
//        cudaMemcpy(d_outputs[sample], Y[sample], sizeof(float) * num_features_outputs, cudaMemcpyHostToDevice);
//    }

    double total = 0;
    std::chrono::duration<float> tt;
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    int turn = 0;

    cout << "Training in coming" << endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
        k = rand() % num_samples;
        Xk = X[k];
        Yk = Y[k];

        host_predict(model, Xk, num_features, isClassification);


        for (int layerID = model->lLenght - 1; layerID >= 0; --layerID) {
            if (layerID == model->lLenght - 1) {
                if (isClassification) {
                    hostCalculateDeltaLayerForClassification(model->DeltaLayer[layerID].vector, model->XLayer->length,
                                                             model->XLayer[layerID].vector, Yk);
                } else {
                    hostCalculateDeltaLayer(model->DeltaLayer[layerID].vector, model->XLayer->length,
                                            model->XLayer[layerID].vector, Yk);
                }
            } else {
                hostCalculateWeightAndDeltaForDelta
                        (
                                model->DeltaLayer[layerID + 1].vector,
                                model->WResLayer[layerID + 1].vector,
                                model->WLayer[layerID + 1].vector,
                                model->WLayer[layerID + 1].nbNeur,
                                model->WLayer[layerID + 1].neurSize
                        );
                hostCalculateSumForDelta
                        (
                                model->WResLayer[layerID + 1].vector,
                                model->WResLayer[layerID + 1].nbNeur,
                                model->WResLayer[layerID + 1].neurSize,
                                model->DeltaLayer[layerID].vector
                        );
                hostCalculateDelta
                        (
                                model->DeltaLayer[layerID].vector,
                                model->XLayer[layerID].vector,
                                model->XLayer[layerID].length
                        );
            }
            if (layerID > 0) {
                hostUpdateWeights
                        (
                                model->WLayer[layerID].vector,
                                model->DeltaLayer[layerID].vector,
                                model->XLayer[layerID].vector,
                                model->WLayer[layerID].nbNeur,
                                model->WLayer[layerID].neurSize,
                                learningRate
                        );
            }
        }
        if (epoch * 100 / epochs != turn) {
            turn = epoch * 100 / epochs;
            if (turn % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                tt = end - start;
                total = (tt).count();
                loading_bar(epoch, epochs, ((total / epoch * epochs) - total));
            }
        }
    }
    loading_bar(epochs, epochs, ((total / epochs * epochs) - total));
    cout << endl;
    cout << "Time passed: " << tt.count() << endl;
}