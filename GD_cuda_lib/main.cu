//
// Created by Carmo on 03/05/2022.
//
#include <iostream>
#include <chrono>
//#include "mlp_model.cuh"
#include <random>
//using namespace Eigen;
using namespace std;

struct Wlayer{
    float* vector;
    int neurSize;
    int nbNeur;
};
struct DeltaLayer {
    float* vector;
    int length;
};
struct Xlayer {
    float* vector;
    int length;
};

struct gpu_Wlayer{
    float* vector;
    int neurSize;
    int nbNeur;
};
struct gpu_DeltaLayer {
    float* vector;
    int length;
};
struct gpu_Xlayer {
    float* vector;
    int length;
};

struct Sumlayer {
    float* vector;
    int length;
};
struct gpu_Sumlayer {
    float* vector;
    int length;
};
struct Mlp_model_cuda {
    int* l;
    int lLenght;
    Xlayer* XLayer;
    Wlayer* WLayer;
    Wlayer* WResLayer;

    gpu_Xlayer* d_XLayer;
    gpu_Wlayer* d_WLayer;
    gpu_Wlayer* d_WResLayer;

    DeltaLayer* DeltaLayer;
    gpu_DeltaLayer* d_DeltaLayer;
    Sumlayer* Sumlayer;
    gpu_Sumlayer* d_Sumlayer;
};
Mlp_model_cuda *create_mlp_model(int* tab,int len) {

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
        cudaMalloc(&model->d_DeltaLayer[layerID].vector, sizeof (float) * nbNeur);

        model->Sumlayer[layerID].length = nbNeur;
        model->Sumlayer[layerID].vector = new float[nbNeur];

        model->d_Sumlayer[layerID].length = nbNeur;
        cudaMalloc(&model->d_Sumlayer[layerID].vector, sizeof (float) * nbNeur);

        for (int elementID = 0; elementID < tab[layerID]; elementID++) {
            model->DeltaLayer[layerID].vector[elementID] = 0;
            model->XLayer[layerID].vector[elementID] = 0;
            model->Sumlayer[layerID].vector[elementID] = 0;
        }

        cudaMemcpy(model->d_XLayer[layerID].vector, model->XLayer[layerID].vector, sizeof(float) * nbNeur, cudaMemcpyHostToDevice);
        cudaMemcpy(model->d_Sumlayer[layerID].vector, model->Sumlayer[layerID].vector, sizeof(float) * nbNeur, cudaMemcpyHostToDevice);

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
                model->WLayer[layerID].vector[elementID] = dist(gen);
                model->WResLayer[layerID].vector[elementID] = 0;
            }
            cudaMemcpy(model->d_WResLayer[layerID].vector, model->WResLayer[layerID].vector, sizeof(float) * neurSize * nbNeur, cudaMemcpyHostToDevice);
            cudaMemcpy(model->d_WLayer[layerID].vector, model->WLayer[layerID].vector, sizeof(float) * nbNeur * neurSize, cudaMemcpyHostToDevice);
        }
    }
    return model;
}
__global__ void calculateLayer(float *W, float *X, float* Wres, int neurSize, int nbNeur) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < neurSize * nbNeur) {
        Wres[i] = W[i] * X[i % neurSize];
    }
}

__device__ void sumNeur(float *X, float* Wres, int neurSize, int nbNeur) {

}

__global__ void sumTanhLayer(float *X, float *Wres, int neurSize, int nbNeur) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbNeur) {
        for (int j = 0; j < neurSize + (i * neurSize) ; j++) {
            X[i] += Wres[j];
        }
        X[i] = tanh(X[i] + 1);
    }
}

__global__ void sumLayer(float *X, float *Wres, int neurSize, int nbNeur) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbNeur) {
        for (int j = 0; j < neurSize + (i * neurSize) ; j++) {
            X[i] += Wres[j];
        }
        X[i] += 1;
    }
}

void hostSumLayer(float *X, float* Wres, int neurSize, int nbNeur) {
    for (int i = 0; i < nbNeur; i++) {
        X[i] = 0;
        for (int j = 0; j < neurSize + (i * neurSize) ; j++) {
            X[i] += Wres[j];
        }
        X[i] += 1;
    }
}
void hostSumTanLayer(float *X, float* Wres, int neurSize, int nbNeur) {
    for (int i = 0; i < nbNeur; i++) {
        X[i] = 0;
        for (int j = 0; j < neurSize + (i * neurSize) ; j++) {
            X[i] += Wres[j];
        }
        X[i] = tanh(X[i] + 1);
    }
}

void hostCalculateLayer(float *W, float *X, float* Wres, int neurSize, int nbNeur) {
    for (int i = 0; i < neurSize * nbNeur; i++) {
        Wres[i] = W[i] * X[i % neurSize];
    }
}

void cuda_predict(Mlp_model_cuda *model, Xlayer X, bool isClassification) {
    std::chrono::duration<float> time;
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    model->XLayer[0] = X;
    cudaMemcpy(model->d_XLayer[0].vector, model->XLayer[0].vector, sizeof(float) * model->XLayer[0].length, cudaMemcpyHostToDevice);
    int neurSize = 0;
    int nbNeur = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int layerID = 1; layerID < model->lLenght; layerID++) {
        nbNeur = model->WLayer[layerID].nbNeur;
        neurSize = model->WLayer[layerID].neurSize;

//        hostCalculateLayer
//            (model->WLayer[layerID].vector,
//               model->XLayer[layerID - 1].vector,
//               model->WResLayer[layerID].vector,
//               model->WLayer[layerID].neurSize,
//                model->WLayer[layerID].nbNeur
//           );

        calculateLayer<<<1, nbNeur * neurSize>>>
        (
            model->d_WLayer[layerID].vector,
            model->d_XLayer[layerID - 1].vector,
            model->d_WResLayer[layerID].vector,
            model->d_WLayer[layerID].neurSize,
            model->d_WLayer[layerID].nbNeur
        );
        if (layerID == (model->lLenght - 1) && !isClassification ) {
//            hostSumLayer(
//                    model->XLayer[layerID].vector,
//                    model->WResLayer[layerID].vector,
//                    neurSize,
//                    nbNeur
//            );
            sumLayer<<<1,nbNeur>>>
            (
                    model->d_XLayer[layerID].vector,
                    model->d_WResLayer[layerID].vector,
                    neurSize,
                    nbNeur
            );
        }
        else {
//            hostSumTanLayer(
//                    model->XLayer[layerID].vector,
//                    model->WResLayer[layerID].vector,
//                    neurSize,
//                    nbNeur
//            );
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
    cudaMemcpy(model->XLayer[model->lLenght - 1].vector, model->d_XLayer[model->lLenght - 1].vector, sizeof(float) * model->d_XLayer[model->lLenght - 1].length, cudaMemcpyDeviceToHost);

}
float** getMatrixXfFromLineMatrix (float* lineMatrix, int nbRows, int nbCols) {
    float** matrix = new float*[nbRows];
    for (int i = 0; i < nbRows; i++) {
        float* vector = new float[nbCols];
        for (int j = 0; j < nbCols; j++) {
            vector[j] = lineMatrix[i * nbCols + j];
        }
        matrix[i] = vector;
    }
    return matrix;
}
void cuda_train(Mlp_model_cuda *model, float* all_samples_inputs, int num_sample, int num_features, float* all_samples_expected_outputs, int num_sample_outputs, int num_features_outputs, int epoch, float learningRate, bool isClassification) {
    auto matrix = getMatrixXfFromLineMatrix(all_samples_inputs, num_sample, num_features);

    for (int sample = 0; sample < num_sample; sample++) {

    }
}

int main() {
//    srand((unsigned int) time(0));
    int len = 4;
    int* tab = new int[4]{10, 10, 10, 2};

    try {
        Mlp_model_cuda *model = create_mlp_model(tab, len);
        auto length = model->XLayer[model->lLenght - 1].length;
        auto lastX = model->XLayer[model->lLenght - 1].vector;

        int nbFeatures = 3;
        int nbSamples = 6;
        //create a random linear matrix
        float* W = new float[nbSamples * nbFeatures];
        for (int i = 0; i < nbSamples * nbFeatures; i++) {
            W[i] = (float) rand() / RAND_MAX;
        }

        auto matrix = getMatrixXfFromLineMatrix(W, nbSamples, nbFeatures);

        for (int i = 0; i < nbSamples; i++) {
            for (int j = 0; j < nbFeatures; j++) {
                cout << matrix[i][j] << " ";
            }
            cout << endl;
        }

        delete model;
        cout << "end" <<  endl;
    }
    catch (const char* message) {
        cout << message <<  endl;
    }
    return 0;
}
