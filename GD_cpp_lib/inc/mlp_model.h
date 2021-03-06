//
// Created by Alexandre Carmone on 02/04/2022.
//
#include <Eigen>
#include <iostream>
#include "../Eigen/Core"

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
    Eigen::VectorX<Eigen::VectorXf> Delta;
    Eigen::VectorX<Eigen::VectorXf> Sum;
    Eigen::MatrixXf res;
};

DLLEXPORT Mlp_model *create_mlp_model(int tab[], int len);

DLLEXPORT float *predict_for_dll(Mlp_model *model, float *sample_inputs, int32_t num_features, bool isClassification);

DLLEXPORT void train_mlp_model(Mlp_model *model, float *all_samples_inputs, int32_t num_sample, int32_t num_features,
                               float *all_samples_expected_outputs, int32_t num_outputs, int32_t num_output_features, float learningRate, int32_t epochs,
                               int32_t isClassification);
DLLEXPORT void delete_mlp_model(Mlp_model *model);

DLLEXPORT void save_model(Mlp_model *model, const char *filename);
DLLEXPORT Mlp_model *load_model(const char *filename);
}

void loading_bar(int i, int n);
void predict(Mlp_model *model, Eigen::VectorXf X, bool isClassification);
Eigen::VectorXd split(const std::string &s, char delim);
std::vector<std::string> split_string(const std::string &s, char delim);
#endif GAMEDETECTION_MLP_MODEL_H
