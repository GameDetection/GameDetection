#include <random>
#include <iostream>
#include "../mlp_model.cuh"
#include <chrono>

using namespace std;


Mlp_model_cuda *create_mlp_model(int32_t *tab, int32_t len) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-0.99, 0.99);

    auto *model = new Mlp_model_cuda;
    model->l = new int[len];
    model->lLenght = len;

    model->WLayer = new Wlayer[len];
    model->XLayer = new Xlayer[len];
    model->WResLayer = new Wlayer[len];
    model->DeltaLayer = new DeltaLayer[len];
    model->Sumlayer = new Sumlayer[len];

    model->d_WLayer = new gpu_Wlayer[len];
    model->d_XLayer = new gpu_Xlayer[len];
    model->d_WResLayer = new gpu_Wlayer[len];
    model->d_DeltaLayer = new gpu_DeltaLayer[len];
    model->d_Sumlayer = new gpu_Sumlayer[len];

    for (int layerID = 0; layerID < len; layerID++) {
        int neurSize, nbNeur = tab[layerID];

        model->l[layerID] = tab[layerID];

        model->XLayer[layerID].length = nbNeur;
        model->XLayer[layerID].vector = new float[nbNeur];

        model->d_XLayer[layerID].length = nbNeur;
        cudaMalloc(&model->d_XLayer[layerID].vector, sizeof(float) * nbNeur);

        model->DeltaLayer[layerID].length = nbNeur;
        model->DeltaLayer[layerID].vector = new float[nbNeur];

        model->d_DeltaLayer[layerID].length = nbNeur;
        cudaMalloc(&model->d_DeltaLayer[layerID].vector, sizeof(float) * nbNeur);

        model->Sumlayer[layerID].length = nbNeur;
        model->Sumlayer[layerID].vector = new float[nbNeur];

        model->d_Sumlayer[layerID].length = nbNeur;
        cudaMalloc(&model->d_Sumlayer[layerID].vector, sizeof(float) * nbNeur);

        for (int elementID = 0; elementID < tab[layerID]; elementID++) {
            model->DeltaLayer[layerID].vector[elementID] = 0;
            model->XLayer[layerID].vector[elementID] = 0;
            model->Sumlayer[layerID].vector[elementID] = 0;
        }

        cudaMemcpy(model->d_XLayer[layerID].vector, model->XLayer[layerID].vector, sizeof(float) * nbNeur,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(model->d_Sumlayer[layerID].vector, model->Sumlayer[layerID].vector, sizeof(float) * nbNeur,
                   cudaMemcpyHostToDevice);

        if (layerID > 0) {
            neurSize = tab[layerID - 1];

            model->WLayer[layerID].nbNeur = nbNeur;
            model->WLayer[layerID].neurSize = neurSize;
            model->WLayer[layerID].vector = new float[tab[layerID] * neurSize];

            model->d_WLayer[layerID].nbNeur = nbNeur;
            model->d_WLayer[layerID].neurSize = neurSize;
            cudaMalloc(&model->d_WLayer[layerID].vector, sizeof(float) * neurSize * nbNeur);

            model->WResLayer[layerID].nbNeur = nbNeur;
            model->WResLayer[layerID].neurSize = neurSize;
            model->WResLayer[layerID].vector = new float[nbNeur * neurSize];

            model->d_WResLayer[layerID].nbNeur = nbNeur;
            model->d_WResLayer[layerID].neurSize = neurSize;
            cudaMalloc(&model->d_WResLayer[layerID].vector, sizeof(float) * neurSize * nbNeur);

            for (int elementID = 0; elementID < nbNeur * neurSize; elementID++) {
                model->WLayer[layerID].vector[elementID] = /*dist(gen)*/3;
                model->WResLayer[layerID].vector[elementID] = 0;
            }
            cudaMemcpy(model->d_WResLayer[layerID].vector, model->WResLayer[layerID].vector,
                       sizeof(float) * neurSize * nbNeur, cudaMemcpyHostToDevice);
            cudaMemcpy(model->d_WLayer[layerID].vector, model->WLayer[layerID].vector,
                       sizeof(float) * nbNeur * neurSize, cudaMemcpyHostToDevice);
        }
    }
    cout << "model ready" << endl;
    return model;
}

__global__ void calculateLayer(float *W, float *X, float *Wres, int neurSize, int nbNeur) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < neurSize * nbNeur) {
        Wres[i] = W[i] * X[i % neurSize];
    }
}


__global__ void sumTanhLayer(float *X, float *Wres, int neurSize, int nbNeur) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbNeur) {
        for (int j = 0; j < neurSize + (i * neurSize); j++) {
            X[i] += Wres[j];
        }
        X[i] = tanh(X[i] + 1);
    }
}

__global__ void sumLayer(float *X, float *Wres, int neurSize, int nbNeur) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbNeur) {
        for (int j = 0; j < neurSize + (i * neurSize); j++) {
            X[i] += Wres[j];
        }
        X[i] += 1;
    }
}

__global__ void calculateDeltaLayerForClassification(float *Delta, int nbNeur, float *X, float *Y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbNeur) {
        Delta[i] = (1 - X[i] * X[i]) * (X[i] - Y[i]);
    }
}

__global__ void calculateDeltaLayer(float *Delta, int nbNeur, float *X, float *Y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbNeur) {
        Delta[i] = X[i] - Y[i];
    }
}

__global__ void calculateWeightAndDeltaForDelta(float *Delta, float *resLayer, float *W, int nbNeur, int neurSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbNeur * neurSize) {
        resLayer[i] = Delta[i % nbNeur] * W[i];
    }
}

__global__ void calculateSumForDelta(float *resLayer, int nbNeur, int neurSize, float *Delta) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbNeur) {
        Delta[i] = 0;
        for (int j = 0; j < neurSize + (i * neurSize); j++) {
            Delta[i] += resLayer[j];
        }
    }
}

__global__ void calculateDelta(float *Delta, float *X, int nbNeur) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbNeur) {
        Delta[i] = (1 - X[i] * X[i]) * Delta[i];
    }
}

__global__ void updateWeights(float *W, float *Delta, float *X, int nbNeur, int neurSize, float learningRate) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbNeur * neurSize) {
        W[i] -= Delta[i % nbNeur] * X[i] * learningRate;
    }
}


float *predict(Mlp_model_cuda *model, float *X, int32_t lenght, bool isClassification) {

    std::chrono::duration<float> time;
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    model->XLayer[0].vector = X;

    cudaMemcpy(model->d_XLayer[0].vector, model->XLayer[0].vector, sizeof(float) * model->XLayer[0].length,
               cudaMemcpyHostToDevice);
    int neurSize = 0;
    int nbNeur = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int layerID = 1; layerID < model->lLenght; layerID++) {
        nbNeur = model->WLayer[layerID].nbNeur;
        neurSize = model->WLayer[layerID].neurSize;


        calculateLayer<<<1, neurSize * nbNeur>>>
                (
                        model->d_WLayer[layerID].vector,
                        model->d_XLayer[layerID - 1].vector,
                        model->d_WResLayer[layerID].vector,
                        model->d_WLayer[layerID].neurSize,
                        model->d_WLayer[layerID].nbNeur
                );
        if (layerID == (model->lLenght - 1) && !isClassification) {
            sumLayer<<<1, nbNeur>>>
                    (
                            model->d_XLayer[layerID].vector,
                            model->d_WResLayer[layerID].vector,
                            neurSize,
                            nbNeur
                    );
        } else {
            sumTanhLayer<<<1, nbNeur>>>
                    (
                            model->d_XLayer[layerID].vector,
                            model->d_WResLayer[layerID].vector,
                            neurSize,
                            nbNeur
                    );
        }
    }
    end = std::chrono::high_resolution_clock::now();
    time = end - start;
    cout << "Time: " << time.count() << endl;
    cudaMemcpy(model->XLayer[model->lLenght - 1].vector, model->d_XLayer[model->lLenght - 1].vector,
               sizeof(float) * model->d_XLayer[model->lLenght - 1].length, cudaMemcpyDeviceToHost);

    return model->XLayer[model->lLenght - 1].vector;
}

void cuda_predict(Mlp_model_cuda *model, float *X, int num_features, bool isClassification) {

    model->d_XLayer[0].vector = X;
    model->d_XLayer[0].length = num_features;

    int neurSize = 0;
    int nbNeur = 0;


    for (int layerID = 1; layerID < model->lLenght; layerID++) {
        nbNeur = model->WLayer[layerID].nbNeur;
        neurSize = model->WLayer[layerID].neurSize;
        calculateLayer<<<nbNeur, neurSize>>>
                (
                        model->d_WLayer[layerID].vector,
                        model->d_XLayer[layerID - 1].vector,
                        model->d_WResLayer[layerID].vector,
                        model->d_WLayer[layerID].neurSize,
                        model->d_WLayer[layerID].nbNeur
                );
        if (layerID == (model->lLenght - 1) && !isClassification) {
            sumLayer<<<1, nbNeur>>>
                    (
                            model->d_XLayer[layerID].vector,
                            model->d_WResLayer[layerID].vector,
                            neurSize,
                            nbNeur
                    );
        } else {
            sumTanhLayer<<<1, nbNeur>>>
                    (
                            model->d_XLayer[layerID].vector,
                            model->d_WResLayer[layerID].vector,
                            neurSize,
                            nbNeur
                    );
        }
    }
}

float **getMatrixXfFromLineMatrix(float *lineMatrix, int nbRows, int nbCols) {
    float **matrix = new float *[nbRows];
    for (int i = 0; i < nbRows; i++) {
        float *vector = new float[nbCols];
        for (int j = 0; j < nbCols; j++) {
            vector[j] = lineMatrix[i * nbCols + j];
        }
        matrix[i] = vector;
    }
    return matrix;
}

void loading_bar(int i, int n, double time) {

    std::cout << "\r";
    std::cout << "[";
    int pos = i * 50 / n;
    for (int j = 0; j < 50; j++) {
        if (j < pos)
            std::cout << "=";
        else if (j == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(i * 100.0 / n) << " % " << time << "s remaining      ";
    std::cout.flush();
}

void cuda_train(Mlp_model_cuda *model, float *all_samples_inputs, int32_t num_samples, int32_t num_features,
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
    for (int sample = 0; sample < num_samples; sample++) {
        cudaMalloc(&(d_inputs[sample]), sizeof(float) * num_features);
        cudaMemcpy(d_inputs[sample], X[sample], sizeof(float) * num_features, cudaMemcpyHostToDevice);

        cudaMalloc(&(d_outputs[sample]), sizeof(float) * num_features_outputs);
        cudaMemcpy(d_outputs[sample], Y[sample], sizeof(float) * num_features_outputs, cudaMemcpyHostToDevice);
    }
    cout << "Data transfer in G-RAM" << endl;

    double total = 0;
    std::chrono::duration<float> tt;
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    int turn = 0;

    cout << "Training in coming" << endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
        k = rand() % num_samples;
        Xk = d_inputs[k];
        Yk = d_outputs[k];

        cuda_predict(model, Xk, num_features, isClassification);


        for (int layerID = model->lLenght - 1; layerID >= 0; --layerID) {
            if (layerID == model->lLenght - 1) {
                if (isClassification) {
                    calculateDeltaLayerForClassification<<<1, model->d_XLayer->length>>>(
                            model->d_DeltaLayer[layerID].vector, model->d_XLayer->length,
                            model->d_XLayer[layerID].vector, Yk);
                } else {
                    calculateDeltaLayer<<<1, model->d_XLayer->length>>>(model->d_DeltaLayer[layerID].vector,
                                                                        model->d_XLayer->length,
                                                                        model->d_XLayer[layerID].vector, Yk);
                }
            } else {
                calculateWeightAndDeltaForDelta<<<model->d_WLayer[layerID + 1].nbNeur, model->d_WLayer[layerID +
                                                                                                       1].neurSize>>>
                        (
                                model->d_DeltaLayer[layerID + 1].vector,
                                model->d_WResLayer[layerID].vector,
                                model->d_WLayer[layerID + 1].vector,
                                model->d_WLayer[layerID + 1].nbNeur,
                                model->d_WLayer[layerID + 1].neurSize
                        );
                calculateSumForDelta<<<1, model->d_WLayer[layerID + 1].nbNeur>>>
                        (
                                model->d_DeltaLayer[layerID + 1].vector,
                                model->d_WLayer[layerID + 1].nbNeur,
                                model->d_WLayer[layerID + 1].neurSize,
                                model->d_DeltaLayer[layerID].vector
                        );
                calculateDelta<<<1, model->d_XLayer[layerID].length>>>
                        (
                                model->d_DeltaLayer[layerID].vector,
                                model->d_XLayer[layerID].vector,
                                model->d_XLayer[layerID].length
                        );
            }
            if (layerID > 0) {
                updateWeights<<<model->d_WLayer[layerID].nbNeur, model->d_WLayer[layerID].neurSize>>>
                        (
                                model->d_WLayer[layerID].vector,
                                model->d_DeltaLayer[layerID].vector,
                                model->d_XLayer[layerID].vector,
                                model->d_WLayer[layerID].nbNeur,
                                model->d_WLayer[layerID].neurSize,
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


