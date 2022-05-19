//
// Created by Alexandre Carmone on 02/04/2022.
//
#include "Eigen/Core"
#include <cstdint>

#ifndef GAMEDETECTION_MLP_MODEL_H
#define GAMEDETECTION_MLP_MODEL_H

#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {

struct Mlp_model {
    Eigen::VectorXd l;
    int inputSize;
    Eigen::VectorX<Eigen::VectorXf> X;
    Eigen::VectorX<Eigen::VectorX<Eigen::VectorXf>> W;
    Eigen::VectorX<Eigen::VectorX<float>> Delta;
    Eigen::VectorX<Eigen::VectorXf> Sum;
    Eigen::MatrixXf res;
};
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
struct Sumlayer {
    float* vector;
    int length;
};
struct Mlp_model_cuda {
    int* l;
    int lLenght;
    Xlayer* XLayer;
    Wlayer* WLayer;
    Wlayer* WResLayer;

    Xlayer* p_XLayer;
    Wlayer* p_WLayer;
    Wlayer* p_WResLayer;

    DeltaLayer* Deltalayer;
    Sumlayer* Sum;
};

//DLLEXPORT Mlp_model *create_mlp_model(int tab[], int len);

DLLEXPORT float *predict_for_dll(Mlp_model *model, float *sample_inputs, int32_t num_features, bool isClassification);

DLLEXPORT void train_mlp_model(Mlp_model *model, float *all_samples_inputs, int32_t num_sample, int32_t num_features,
                               float *all_samples_expected_outputs, int32_t num_outputs, float learningRate, int32_t epochs,
                               int32_t isClassification);
DLLEXPORT void delete_mlp_model(Mlp_model *model);

DLLEXPORT void save_model_to_csv(Mlp_model *model, const char *filename);
DLLEXPORT Mlp_model *load_model_from_csv(const char *filename);
}

void loading_bar(int i, int n);
void predict(Mlp_model *model, Eigen::VectorXf X, bool isClassification);
Eigen::VectorXd split(const std::string &s, char delim);
std::vector<std::string> split_string(const std::string &s, char delim);


#endif GAMEDETECTION_MLP_MODEL_H
