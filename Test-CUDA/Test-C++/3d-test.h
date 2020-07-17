#include<iostream>
#include "launcher.cuh"

class pencilComputation {
    public:

    int inputMatrix[10][10][10];
    int outputMatrix[10][10][10];
    int pencilVector[10];
    
    //Function to perform pencil computation
    void pencilComputationFunction();
    //Function to extract 1-D vector from inputMatrix
    void collectPencilVector();
    //Function to scale the 1-D vector
    void scalingFunction(int[],int);
    //Function to distribute the scaled 1-D vector to outputMatrix
    void distributeFunction(int[],int,int);
};
