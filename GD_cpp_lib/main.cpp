#include <iostream>
#include "Eigen"
//#include "inc/mlp_model.h"
#include "inc/mlp_modelV2.h"
//#include "../GD_cpp_linear_model/linear_model.h"
#include <chrono>


using namespace Eigen;
using namespace std;



int main() {


    srand((unsigned int) time(0));
    //calculate the time of the program


    // Initialize the model
    int nlp[3] = {2,  2,1};
    Mlp_model *model = create_mlp_model(nlp, 3);

    // Initialize the data
    // The data is a matrix of size (n, m) where n is the number of samples and m is the number of features
    //X is a linear matrix of size (n, m)
    //Y is a linear matrix of size (n)
    //malloc data here

    float X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float Y[4][1] = {{0}, {1}, {1}, {0}};

    //create a linear matrix of size (n, m)
    int n = 4;
    int m = 2;
    float X1[8];
    float Y1[4];
    for(int i = 0; i < n; i++){
        Y1[i] = Y[i][0];
        for(int j = 0; j < m; j++){
            X1[i*m+j] = X[i][j];
        }
    }
//    cout << "On parcours les couches pour voir les poids" << endl;
//    for (int layer = 1; layer < model->W.size(); ++layer) {
//        for (int neur = 0; neur < model->W(layer).size(); ++neur) {
//            cout << "W" << layer << "," << neur << ": " << model->W(layer)(neur) << endl;
//        }
//    }
//    cout << "On parcours les couches pour voir les valeurs" << endl;
//    for (int layer = 0; layer < model->X.size(); ++layer) {
//
//        cout << "X" << layer << " : " << model->X(layer) << endl;
//    }
    predict_for_dll(model, X[0], 2, 1);
    cout << "res: " << endl << model->X(2) << endl;

    train_mlp_model(model, X1, 4, 2, Y1, 4,1, 0.01, 100000, 1);
    cout << endl;
//    for (int sample = 0; sample < n; ++sample) {
//        predict_for_dll(model, X[sample], 2, 1);
//        cout << "G(Xk): " << model->X(2)(0)  << endl << "Yk: " << Y[sample][0] << endl;
//    }
    cout << "On parcours les couches pour voir les poids" << endl;
    for (int layer = 1; layer < model->W.size(); ++layer) {
        for (int neur = 0; neur < model->W(layer).size(); ++neur) {
            cout << "W" << layer << "," << neur << ": " << model->W(layer)(neur) << endl;
        }
    }
//    std::cout << "Time taken by function: " << diff.count() << " seconds" << std::endl;

    save_model_to_csv(model, "C:/Users/Carmo/OneDrive/Documents/ESGI/projet_annuel/GameDetection/GD_cpp_lib/model.csv");


    model = load_model_from_csv("C:/Users/Carmo/OneDrive/Documents/ESGI/projet_annuel/GameDetection/GD_cpp_lib/model.csv");
    cout << "On parcours les couches pour voir les poids" << endl;
    for (int layer = 1; layer < model->W.size(); ++layer) {
        for (int neur = 0; neur < model->W(layer).size(); ++neur) {
            cout << "W" << layer << "," << neur << ": " << model->W(layer)(neur) << endl;
        }
    }
    cout << endl;
//    for (int sample = 0; sample < n; ++sample) {
//        predict_for_dll(model, X[sample], 2, 1);
//        cout << "G(Xk): " << model->X(2)(0)  << endl << "Yk: " << Y[sample][0] << endl;
//    }
//    save_model_to_csv(model, "C:/Users/Carmo/OneDrive/Documents/ESGI/projet_annuel/GameDetection/GD_cpp_lib/model.csv");

    return 0;
}



