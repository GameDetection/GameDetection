//
// Created by Alexandre Carmone on 02/04/2022.
//
#include <cstdint>

#ifndef GAMEDETECTION_MLP_MODEL_H
#define GAMEDETECTION_MLP_MODEL_H

#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {

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

DLLEXPORT Mlp_model_cuda *create_mlp_model(int* tab,int len);

DLLEXPORT void predict(Mlp_model_cuda *model, Xlayer X, bool isClassification);

DLLEXPORT void cuda_train(Mlp_model_cuda *model, float* all_samples_inputs, int num_samples, int num_features, float* all_samples_expected_outputs, int num_samples_outputs, int num_features_outputs, int epochs, float learningRate, bool isClassification);

//DLLEXPORT void save_model_to_csv(Mlp_model *model, const char *filename);
//DLLEXPORT Mlp_model *load_model_from_csv(const char *filename);
}
void cuda_predict(Mlp_model_cuda *model, float* X, int num_features, bool isClassification);

void hostSumLayer(float *X, float* Wres, int neurSize, int nbNeur);
void hostSumTanLayer(float *X, float* Wres, int neurSize, int nbNeur);
void hostCalculateLayer(float *W, float *X, float* Wres, int neurSize, int nbNeur);


__global__ void calculateLayer(float *W, float *X, float* Wres, int neurSize, int nbNeur);
__global__ void sumTanhLayer(float *X, float *Wres, int neurSize, int nbNeur);
__global__ void sumLayer(float *X, float *Wres, int neurSize, int nbNeur);
__global__ void calculateDeltaLayerForClassification(float* Delta, int nbNeur, float* X, float* Y);
__global__ void calculateDeltaLayer(float* Delta, int nbNeur, float* X, float* Y);
__global__ void calculateWeightAndDeltaForDelta(float* Delta, float* resLayer, float* W, int nbNeur, int neurSize);
__global__ void calculateSumForDelta(float*  resLayer, int nbNeur, int neurSize, float* Delta);
__global__ void calculateDelta(float* Delta, float* X, int nbNeur);
__global__ void updateWeights(float* W, float* Delta, float* X, int nbNeur, int neurSize, float learningRate);

#endif GAMEDETECTION_MLP_MODEL_H
