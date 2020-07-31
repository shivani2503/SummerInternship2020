#include "3d-test.cuh"
#include<iostream>
#include<stdio.h>

__host__ __device__
 void scalingFunction(int array[]) {
        

    for(int i=0; i<8; i++) {
        array[i] = array[i] * 2.0;
    }


}

__host__ __device__
void  distributeFunction(pencilComputation& p1,int x,int y){
  
      for(int z=0; z<8; z++) {
        p1.outputMatrix[x][y][z] = p1.pencilVector[z];
      }

}

__host__ __device__
void tOPrint(int i,int j,int k) {

 //pencilComputation p2;  
  printf("The value of i:%d,  j:%d, inputMatrix:%d, pencilVector:%d",i,j);//,p2.inputMatrix[i][j][k],p2.pencilVector[k]);

}

__global__ 
void pencilComputationFunction(pencilComputation& p1){
 
//pencilComputation p1;
//  auto p1 = *p1_in; // dereference the pointer
  p1.DStats("At start of pencilComputationFunction call");

  int i = threadIdx.x ;//+ blockIdx.x * blockDim.x;
  int j = threadIdx.y ; // + blockIdx.y * blockDim.y;
  int z = threadIdx.z;
  for(int k=0; k<8 ; k++) {
    p1.pencilVector[k] = p1.inputMatrix[i][j][k];
  }
  scalingFunction(p1.pencilVector);
  p1.DStats("After scalingFunction in pencilComputationFunction call");
  distributeFunction(p1,i,j);
  p1.DStats("At END of pencilComputationFunction call");
// printf("**************************************\n");

}

void pencilComputation::launcher(){
    
//
//    for(int i=0; i<2; i++) {
//      for(int j=0; j<2; j++) {
//          for(int k=0; k<2; k++) {
//              inputMatrix[i][j][k] = 10;
//          }
//      }
//  }

    dim3 grid(1,1,1);
    dim3 block(2,2,2);

   // #pragma acc enter data copyin(inputMatrix) copyout(outputMatrix)
 // pencilComputationFunction<<<grid,block>>>(*this);
  pencilComputationFunction<<<1,1>>>(*this);
  //pencilComputationFunction<<<1,1>>>();
    cudaDeviceSynchronize();
   

}

__host__ 
void pencilComputation::CStats(const char* message) {
  // Get sum, average and num-nonzero for each of inputMatrix & outputMatrix:
  printf("[CPU] Stats: %s\n", message);
  printf("               Sum      Average      NNZ\n");
  int sumInput = 0;
  int sumOutput = 0;
  int nnzInput = 0;
  int nnzOutput = 0;
  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      for (int k = 0; k < matrix_size; k++) {
        if (inputMatrix[i][j][k] != 0) { nnzInput++; }
        if (outputMatrix[i][j][k] != 0) { nnzOutput++; }
        sumInput += inputMatrix[i][j][k];
        sumOutput += outputMatrix[i][j][k];
      }
    }
  }

  double averageInput  = sumInput  / (double)(matrix_size * matrix_size * matrix_size);
  double averageOutput = sumOutput / (double)(matrix_size * matrix_size * matrix_size);
  printf("  inputMatrix    %d       %g           %d\n", sumInput, averageInput, nnzInput);
  printf("  outputMatrix   %d       %g           %d\n", sumOutput, averageOutput, nnzOutput);

  int sumVector = 0;
  int nnzVector = 0;
  for (int i = 0; i < (matrix_size * matrix_size * matrix_size); i++) {
    sumVector += pencilVector[i];
    if (pencilVector[i] != 0) { nnzVector++; }
  }
  double averageVector = sumVector / (double)(matrix_size * matrix_size * matrix_size);
  printf("  pencilVector   %d       %g           %d\n", sumVector, averageVector, nnzVector);
  printf("\n");
}

__device__
void pencilComputation::DStats(const char* message) {
  // Get sum, average and num-nonzero for each of inputMatrix & outputMatrix:
  printf("[Device] Matrix Stats: %s\n", message);
  printf("               Sum      Average      NNZ\n");
  int sumInput = 0;
  int sumOutput = 0;
  int nnzInput = 0;
  int nnzOutput = 0;
  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      for (int k = 0; k < matrix_size; k++) {
        if (inputMatrix[i][j][k] != 0) { nnzInput++; }
        if (outputMatrix[i][j][k] != 0) { nnzOutput++; }
        sumInput += inputMatrix[i][j][k];
        sumOutput += outputMatrix[i][j][k];
      }
    }
  }

  double averageInput  = sumInput  / (double)(matrix_size * matrix_size * matrix_size);
  double averageOutput = sumOutput / (double)(matrix_size * matrix_size * matrix_size);
  printf("  inputMatrix    %d       %g           %d\n", sumInput, averageInput, nnzInput);
  printf("  outputMatrix   %d       %g           %d\n", sumOutput, averageOutput, nnzOutput);
  int sumVector = 0;
  int nnzVector = 0;
  for (int i = 0; i < (matrix_size * matrix_size * matrix_size); i++) {
    sumVector += pencilVector[i];
    if (pencilVector[i] != 0) { nnzVector++; }
  }
  double averageVector = sumVector / (double)(matrix_size * matrix_size * matrix_size);
  printf("  pencilVector   %d       %g           %d\n", sumVector, averageVector, nnzVector);
  printf("\n");
}
