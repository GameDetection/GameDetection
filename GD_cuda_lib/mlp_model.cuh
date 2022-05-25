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

DLLEXPORT Mlp_model_cuda *create_mlp_model(int32_t* tab, int32_t len);

DLLEXPORT float* predict(Mlp_model_cuda *model, float* X, int32_t lenght, bool isClassification);

DLLEXPORT void cuda_train(Mlp_model_cuda *model, float* all_samples_inputs, int32_t num_samples, int32_t num_features, float* all_samples_expected_outputs, int32_t num_samples_outputs, int32_t num_features_outputs, int32_t epochs, float learningRate, bool isClassification);
DLLEXPORT void train(Mlp_model_cuda *model, float* all_samples_inputs, int32_t num_samples, int32_t num_features, float* all_samples_expected_outputs, int32_t num_samples_outputs, int32_t num_features_outputs, int32_t epochs, float learningRate, bool isClassification);
//DLLEXPORT void save_model_to_csv(Mlp_model *model, const char *filename);
//DLLEXPORT Mlp_model *load_model_from_csv(const char *filename);
}
float** getMatrixXfFromLineMatrix (float* lineMatrix, int nbRows, int nbCols);
void loading_bar(int i, int n, double time);

void cuda_predict(Mlp_model_cuda *model, float* X, int num_features, bool isClassification);
void host_predict(Mlp_model_cuda *model, float* X, int32_t lenght, bool isClassification);

void hostSumLayer(float *X, float* Wres, int neurSize, int nbNeur);
void hostSumTanLayer(float *X, float* Wres, int neurSize, int nbNeur);
void hostCalculateLayer(float *W, float *X, float* Wres, int neurSize, int nbNeur);
void hostUpdateWeights(float* W, float* Delta, float* X, int nbNeur, int neurSize, float learningRate);
void hostCalculateDelta(float* Delta, float* X, int nbNeur);
void hostCalculateSumForDelta(float*  resLayer, int nbNeur, int neurSize, float* Delta);
void hostCalculateWeightAndDeltaForDelta(float* Delta, float* resLayer, float* W, int nbNeur, int neurSize);
void hostCalculateDeltaLayer(float* Delta, int nbNeur, float* X, float* Y);
void hostCalculateDeltaLayerForClassification(float* Delta, int nbNeur, float* X, float* Y);


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
