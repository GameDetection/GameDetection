//
// Created by Alexandre Carmone on 02/04/2022.
//
#include "../inc/mlp_model.h"
#include <iostream>
#include <utility>
#include <cstdint>
#include <fstream>
#include <chrono>

using namespace Eigen;
using namespace std;
/**
 * This function create a mlp model
 * @param tab
 * @param len
 * @return
 */
Mlp_model *create_mlp_model(int tab[], int len) {
    // Create a model with the given parameters
    std::cout << "Creating a model with " << len << " layers" << std::endl;
    auto *model = new Mlp_model();
    model->inputSize = tab[0];
    model->l.resize(len);
    for (int i = 0; i < len; ++i) {
        model->l(i) = tab[i];
    }
    model->X.resize(len);
    model->W.resize(len);
    model->Delta.resize(len);
    model->Sum.resize(len);
    for (int i = 0; i < len; ++i) {
        model->Delta(i).resize(tab[i]);
        model->Delta(i).setZero();

        model->X(i).resize(tab[i]);
        model->X(i).setOnes();
        model->W(i).resize(tab[i]);
        model->Sum(i).resize(tab[i]);
        model->Sum(i).setZero();
        for (int j = 0; j < tab[i]; ++j) {
            if (i > 0) {
                model->W(i)(j).resize(tab[i - 1]);
                model->W(i)(j).setRandom();
            }
        }
    }
    return model;
}

/**
 * This function delete a mlp model
 * @param model
 */
void delete_mlp_model(Mlp_model *model) {
    delete model;
}

/**
 * This function predict the result of the neural network
 * @param model
 * @param X
 * @param isClassification
 * @return
 */
void predict(Mlp_model *model, VectorXf X, bool isClassification) {
    model->X(0) = std::move(X);
    for (int layerID = 1; layerID < model->X.size(); layerID++) {
        for (int neurID = 0; neurID < model->X(layerID).size(); neurID++) {

            if (model->X.size() - 1 == layerID and !isClassification) {
                model->X(layerID)(neurID) = (model->X(layerID - 1).transpose() * model->W(layerID)(neurID)) + 1;
            } else {
                model->X(layerID)(neurID) = tanh((model->X(layerID - 1).transpose() * model->W(layerID)(neurID)) + 1 );
            }
        }
    }
}

/**
 * This function convert a line matrix to a Matrix
 * @param table
 * @param rows
 * @param cols
 * @return
 */
MatrixXf getMatrixXfFromLineMatrix(float table[], int rows, int cols) {
    MatrixXf res;
    res.resize(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i, j) = table[i * cols + j];
        }
    }
    return res;
}

/**
 * This function convert a line matrix to a Matrix
 * @param table
 * @param rows
 * @param cols
 * @return
 */
VectorXf getVectorXfFromLineVector(float table[], int size) {
    VectorXf res;
    res.resize(size);
    res = VectorXf::Map(table, size);
    return res;
}


/**
 * This function predict the result of the neural network for dll implementation
 * @param model
 * @param sample_inputs
 * @param num_features
 * @param isClassification
 * @return
 */
float *predict_for_dll(Mlp_model *model, float *sample_inputs, int32_t num_features, bool isClassification) {

    model->X(0) = getVectorXfFromLineVector(sample_inputs, num_features);
    for (int layerID = 1; layerID < model->X.size(); layerID++) {
        for (int neurID = 0; neurID < model->X(layerID).size(); neurID++) {

            if (model->X.size() - 1 == layerID and !isClassification) {
                model->X(layerID)(neurID) = (model->X(layerID - 1).transpose() * model->W(layerID)(neurID)) + 1;
            } else {
                model->X(layerID)(neurID) = tanh((model->X(layerID - 1).transpose() * model->W(layerID)(neurID)) + 1 );
            }
        }
    }
    float *Z = new float[model->X(model->X.size() - 1).size()];
    //convert the result to a line matrix
    for (int i = 0; i < model->X(model->X.size() - 1).size(); i++) {
        Z[i] = model->X(model->X.size() - 1)(i);
    }

    return Z;
}

/**
 * This function display a loading bar
 * @param i
 * @param n
 */
void loading_bar(int i, int n, double time) {

    std::cout << "\r";
    std::cout << "[";
    int pos = i * 50 / n;
    for (int j = 0; j < 50; j++)
    {
        if (j < pos)
            std::cout << "=";
        else if (j == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << setprecision(3) << "] " << int(i * 100.0 / n) << " % " << time << "s remaining      ";
    std::cout.flush();
}


/**
 * This function train_mlp_model the neural network for dll implementation
 * @param model
 * @param all_samples_inputs
 * @param num_sample
 * @param num_features
 * @param all_samples_expected_outputs
 * @param num_outputs
 * @param learningRate
 * @param epochs
 * @param isClassification
 */
void train_mlp_model(Mlp_model *model, float *all_samples_inputs, int32_t num_sample, int32_t num_features,
                     float *all_samples_expected_outputs, int32_t num_outputs, float learningRate, int32_t epochs,
                     int32_t isClassification) {

    MatrixXf inputs = getMatrixXfFromLineMatrix(all_samples_inputs, num_sample, num_features);
    VectorXf Y = getVectorXfFromLineVector(all_samples_expected_outputs, num_outputs);

    int k = 0;
    VectorXf Xk;
    VectorXf Yk;

    std::chrono::duration<double> diff;
    double total = 0;
    std::chrono::duration<float> tt;
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    int turn = 0;
    for (int i = 0; i <= epochs; ++i) {

        k = rand() % inputs.rows();
        Xk = inputs.row(k);
        Yk = Y.row(k);
        predict(model, Xk, isClassification);
        //On parcours les couche du model
        for (int layerID = model->W.size() - 1; layerID >= 0; --layerID) {
            // Ici on calcule de le Delta de la dernière couche
            if (layerID == model->W.size() - 1) {
                if (isClassification) {
                    //on fait le calcul de la dernière couche en fonction de la classification
                    model->Delta(layerID) = (1 - model->X(layerID).cwiseProduct(model->X(layerID)).array()) *
                                            (model->X(layerID) - Yk).array();
                } else {
                    model->Delta(layerID) = model->X(layerID) - Yk;
                }
            }
                // pour les autres couches
            else {
                model->Sum(layerID).setZero();
                // pour chaque neuronne dans la couche
                for (int neurID = 0; neurID < model->Delta(layerID + 1).size(); ++neurID) {
                    model->Sum(layerID) += model->W(layerID + 1)(neurID) * model->Delta(layerID + 1)(neurID);
                }
                //on calcule le delta de la couche
                model->Delta(layerID) = (1 - model->X(layerID).cwiseProduct(model->X(layerID)).array()) * model->Sum(layerID).array();
            }
            if (layerID > 0) {
                //On fait la mise à jour des poids
                for (int neurID = 0; neurID < model->W(layerID).size(); ++neurID) {
                    model->W(layerID)(neurID) -= learningRate * model->X(layerID - 1) * model->Delta(layerID)(neurID);
                }
            }
        }


        if(i * 100 / epochs != turn) {
            turn = i * 100 / epochs;
            if (turn%2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                tt = end - start;
                total = (tt).count();
                loading_bar(i, epochs, ((total / i) * epochs) - total);
            }
        }
    }

    std::cout << std::endl;
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Time passed " << diff.count() << "s";
}

/**
 * This function save the model in file passed in parameter
 * @param model
 * @param filename
 */
void save_model_to_csv(Mlp_model *model, const char *filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "model: " << model->l.transpose() << std::endl;
        file << "" << endl;
        for (int layerID = 1; layerID < model->W.size(); ++layerID) {
            for (int neurID = 0; neurID < model->W(layerID).size(); ++neurID) {
                for (int element = 0; element < model->W(layerID)(neurID).size(); ++element) {
                    if (element < model->W(layerID)(neurID).size()-1) {
                        file << model->W(layerID)(neurID)(element) << "|";
                    }
                    else {
                        file << model->W(layerID)(neurID)(element);
                    }
                }
                if (neurID < model->W(layerID).size()-1) {
                    file << ";";
                }

            }
            file<<std::endl;
        }
        file.close();
    }
}

//this function split a string with a delimiter a return an Eigen vector of int
VectorXd split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> tokens;
    while (std::getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    VectorXd res(tokens.size()-1);
    for (int i = 1; i < tokens.size(); ++i) {
        res(i-1) = stoi(tokens[i]);
    }
    return res;
}
//This function split a string to delimiter and return a string table
std::vector<std::string> split_string(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> tokens;
    while (std::getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}
/**
 * This function load a mlp model from a file passed in parameter
 * @param filename
 * @return
 */
Mlp_model *load_model_from_csv(const char *filename) {
    auto* model = new Mlp_model;
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
         //on copy la pemière ligne dans la variable tes
        getline(file, line);
        model->l = split(line, ' ');
        int size = model->l.size();

        model->X.resize(size);
        model->W.resize(size);
        model->Delta.resize(size);
        model->Sum.resize(size);
        for (int i = 0; i < size; ++i) {
            getline(file, line);
            model->Delta(i).resize(model->l(i));
            model->Delta(i).setZero();

            model->X(i).resize(model->l(i));
            model->X(i).setOnes();
            model->W(i).resize(model->l(i));
            model->Sum(i).resize(model->l(i));
            model->Sum(i).setZero();
            auto neur = split_string(line, ';');

            if (i > 0) {
                for (int j = 0; j < model->l(i); ++j) {
                    auto elements = split_string(neur[j], '|');
                    model->W(i)(j).resize(model->l(i - 1));
                    for (int elementID = 0; elementID < elements.size(); ++elementID) {
                        model->W(i)(j)(elementID) = stof(elements[elementID]);
                    }
                }
            }
        }
    }
    return model;
}

