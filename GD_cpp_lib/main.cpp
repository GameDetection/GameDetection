#include <iostream>
#include "Eigen/Core"
#include "inc/mlp_model.h"
#include <chrono>



using namespace Eigen;
using namespace std;
//This function multiply the input vector with the weight matrix and then add the bias
MatrixXd multiply_and_add(MatrixXd input, MatrixXd weight, MatrixXd bias)
{
    MatrixXd result = input * weight;
    result = result + bias;
    return result;
}

int main() {


    srand((unsigned int) time(0));
    //calculate the time of the program


    // Initialize the model
    int nlp[3] = {2,3,  1};
    Mlp_model *model = create_mlp_model(nlp, 3);

    // Initialize the data
    // The data is a matrix of size (n, m) where n is the number of samples and m is the number of features
    //X is a linear matrix of size (n, m)
    //Y is a linear matrix of size (n)
    //malloc data here

    float X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float Y[4] = {0, 1, 1, 0};


    train_mlp_model(model, reinterpret_cast<float *>(X), 4, 2, Y, 4, 0.01, 1000000, 1);


//    std::cout << "Time taken by function: " << diff.count() << " seconds" << std::endl;

//    save_model_to_csv(model, "C:/Users/Carmo/OneDrive/Documents/ESGI/projet_annuel/GameDetection/model.csv");

//    model = load_model("C:/Users/Carmo/OneDrive/Documents/ESGI/projet_annuel/GameDetection/model.txt");
    save_model_to_csv(model, "C:/Users/Carmo/OneDrive/Documents/ESGI/projet_annuel/GameDetection/GD_cpp_lib/model.csv");
    model = load_model_from_csv("C:/Users/Carmo/OneDrive/Documents/ESGI/projet_annuel/GameDetection/GD_cpp_lib/model.csv");
    save_model_to_csv(model, "C:/Users/Carmo/OneDrive/Documents/ESGI/projet_annuel/GameDetection/GD_cpp_lib/model.csv");

    return 0;
}



