//
// Created by Carmo on 24/05/2022.
//
#include "Eigen"
#include <iostream>

using namespace Eigen;
#ifndef MLP_MODEL_MLP_MODELV2_H
#define MLP_MODEL_MLP_MODELV2_H
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
    Eigen::VectorX<Eigen::VectorXf> Delta;
    Eigen::VectorX<Eigen::VectorXf> Sum;
    Eigen::MatrixXf res;
};

DLLEXPORT Mlp_model *create_mlp_model(int tab[], int len);
DLLEXPORT void delete_mlp_model(Mlp_model *model);

DLLEXPORT float *predict_for_dll(Mlp_model *model, float *sample_inputs, int32_t num_features, bool isClassification);

DLLEXPORT void train_mlp_model(Mlp_model *model, float *all_samples_inputs, int32_t num_sample, int32_t num_features,
                               float *all_samples_expected_outputs, int32_t num_outputs, int32_t num_output_features,
                               float learningRate, int32_t epochs,
                               int32_t isClassification);

DLLEXPORT void save_model(Mlp_model *model, char *filename);
DLLEXPORT Mlp_model *load_model(const char *filename);
}

void loading_bar(int i, int n, double time);

VectorXf getVectorXfFromLineVector(float table[], int size);

MatrixXf getMatrixXfFromLineMatrix(float table[], int rows, int cols);

#endif MLP_MODEL_MLP_MODELV2_H
