//
// Created by Carmo on 24/05/2022.
//

#include "../inc/mlp_modelV2.h"
#include <chrono>

using namespace Eigen;
using namespace std;
/**
 * This function creates a new model
 * @param tab
 * @param len
 * @return
 */
Mlp_model *create_mlp_model(int tab[], int len) {
    std::cout << "Creating a model with " << len << " layers" << std::endl;
    auto *model = new Mlp_model();
    model->l.resize(len);
    for (int i = 0; i < len; ++i) {
        model->l(i) = tab[i];
    }
    model->X.resize(len);
    model->W.resize(len);
    model->Delta.resize(len);
    model->Sum.resize(len);

    for (int i = 0; i < len; ++i) {
        model->Delta(i).resize(tab[i] + 1);
        model->Delta(i).setZero();

        model->X(i).resize(tab[i] + 1);
        model->X(i).setOnes();

        model->W(i).resize(tab[i]);
        model->Sum(i).resize(tab[i] + 1);
        model->Sum(i).setZero();
        for (int j = 0; j < tab[i]; ++j) {
            if (i > 0) {
                model->W(i)(j).resize(tab[i - 1] + 1);
                model->W(i)(j).setRandom();
            }
        }
    }
    return model;
}
/**
 * This function delete the model
 * @param model
 */
void delete_mlp_model(Mlp_model *model) {
    delete model;
}
/**
 * This function is used to predict outside the training process
 * @param model
 * @param sample_inputs
 * @param num_features
 * @param isClassification
 * @return
 */
float *predict_for_dll(Mlp_model *model, float *sample_inputs, int32_t num_features, bool isClassification) {

    model->X(0) = getVectorXfFromLineVector(sample_inputs, num_features);
    for (int layerID = 1; layerID < model->W.size(); layerID++) {
        for (int neurID = 0; neurID < model->W(layerID).size(); neurID++) {

            if (model->X.size() - 1 == layerID and !isClassification) {
                model->X(layerID)(neurID) = (model->W(layerID)(neurID).transpose() * model->X(layerID - 1));
            } else {
                model->X(layerID)(neurID) = tanh((model->W(layerID)(neurID).transpose() * model->X(layerID - 1)));
            }
        }
    }
    float *Z = new float[model->X(model->X.size() - 1).size() - 1];
    //convert the result to a line matrix
    for (int i = 0; i < model->X(model->X.size() - 1).size() - 1; i++) {
        Z[i] = model->X(model->X.size() - 1)(i);
    }

    return Z;
}
/**
 * This function is used to make a prediction for the training
 * @param model
 * @param X
 * @param isClassification
 */
void predict(Mlp_model *model, VectorXf X, bool isClassification) {
    model->X(0) = std::move(X);
    for (int layerID = 1; layerID < model->W.size(); layerID++) {
        for (int neurID = 0; neurID < model->W(layerID).size(); neurID++) {

            if (model->X.size() - 1 == layerID and !isClassification) {
                model->X(layerID)(neurID) = (model->W(layerID)(neurID).transpose() * model->X(layerID - 1));
            } else {
                model->X(layerID)(neurID) = tanh((model->W(layerID)(neurID).transpose() * model->X(layerID - 1)));
            }
        }
    }
}
/**
 * This function is used to train the model
 * @param model
 * @param all_samples_inputs
 * @param num_sample
 * @param num_features
 * @param all_samples_expected_outputs
 * @param num_outputs
 * @param num_output_features
 * @param learningRate
 * @param epochs
 * @param isClassification
 */
void train_mlp_model(Mlp_model *model, float *all_samples_inputs, int32_t num_sample, int32_t num_features,
                     float *all_samples_expected_outputs, int32_t num_outputs, int32_t num_output_features,
                     float learningRate, int32_t epochs,
                     int32_t isClassification) {

    MatrixXf inputs = getMatrixXfFromLineMatrix(all_samples_inputs, num_sample, num_features);
    MatrixXf Y = getMatrixXfFromLineMatrix(all_samples_expected_outputs, num_outputs, num_output_features);

//    std::cout << "inputs: " << endl << inputs << std::endl;
    cout << "Training the model" << endl;

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
                for (int neurID = 0; neurID < model->W(layerID + 1).size(); ++neurID) {
                    model->Sum(layerID) += model->W(layerID + 1)(neurID) * model->Delta(layerID + 1)(neurID);
                }
                //on calcule le delta de la couche
                model->Delta(layerID) =
                        (1 - model->X(layerID).cwiseProduct(model->X(layerID)).array()) * model->Sum(layerID).array();
            }
            if (layerID > 0) {
                //On fait la mise à jour des poids
                for (int neurID = 0; neurID < model->W(layerID).size(); ++neurID) {
                    model->W(layerID)(neurID) -= learningRate * model->X(layerID - 1) * model->Delta(layerID)(neurID);
                }
            }
        }


        if (i * 100 / epochs != turn) {
            turn = i * 100 / epochs;
            if (turn % 2 == 0) {
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
    std::cout << "Time passed " << diff.count() << "s" << endl;
}
/**
 * This function display a loading bar with the time prediction
 * @param i
 * @param n
 * @param time
 */
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
    std::cout << setprecision(3) << "] " << int(i * 100.0 / n) << " % " << time << "s remaining      ";
    std::cout.flush();
}

/**
 * This function is used to convert a float table to a Eigen Vector
 * @param table
 * @param size
 * @return
 */
VectorXf getVectorXfFromLineVector(float table[], int size) {
    VectorXf res;
    res.resize(size + 1);
    res = VectorXf::Map(table, size + 1);
    res(size) = 1;
    return res;
}
/**
 * This function is used to convert a line matrix (table of float) to a Eigen matrix
 * @param table
 * @param rows
 * @param cols
 * @return
 */
MatrixXf getMatrixXfFromLineMatrix(float table[], int rows, int cols) {
    MatrixXf res;
    res.resize(rows, cols + 1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res(i, j) = table[i * cols + j];
        }
        res(i, cols) = 1;
    }
    return res;
}