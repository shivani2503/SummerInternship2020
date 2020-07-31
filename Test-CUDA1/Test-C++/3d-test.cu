#include "3d-test.cuh"
#include<iostream>
#include<stdio.h>

__host__ __device__
 void scalingFunction(int array[],int x) {
        

    for(int i=0; i<8; i++) {
        array[i] = array[i] * 2;
    }


}

__host__ __device__
void  distributeFunction(int array[],int x,int y){
      //pencilComputation p2; 
 
      for(int z=0; z<8; z++) {
          
        //  p2.outputMatrix[x][y][z] = p2.pencilVector[z];
      }

}

__host__ __device__
void tOPrint(int i,int j,int k) {

 pencilComputation p2;  
  printf("The value of i:%d,  j:%d, inputMatrix:%d, pencilVector:%d",i,j);//,p2.inputMatrix[i][j][k],p2.pencilVector[k]);

}

__global__ 
void pencilComputationFunction(){
 
pencilComputation p1;
   
int i = threadIdx.x ;//+ blockIdx.x * blockDim.x;
   int j = threadIdx.y ; // + blockIdx.y * blockDim.y;
    int z = threadIdx.z;
   for(int k=0; k<8 ; k++) {
 
  
    printf("The value of i: %d ,  j: %d ,z: %d, inputMatrix Value: %d\n",i,j,z,p1.inputMatrix[i][j][k]);//,p1.inputMatrix[i][j][k],p1.pencilVector[k]);
     printf("**************************************\n");

   // tOPrint(i,j,k);
    // p1.pencilVector[k] = p1.inputMatrix[i][j][k];

   
   // scalingFunction(p1.pencilVector,i);
   // distributeFunction(p1.pencilVector,i,j);
 }
// printf("**************************************\n");

}

void pencilComputation::launcher(){
    
    for(int i=0; i<2; i++) {
      for(int j=0; j<2; j++) {
          for(int k=0; k<2; k++) {
              inputMatrix[i][j][k] = 10;
          }
      }
  }


    dim3 grid(1,1,1);
    dim3 block(2,2,2);

   // #pragma acc enter data copyin(inputMatrix) copyout(outputMatrix)
    pencilComputationFunction<<<grid,block>>>();
   // pencilComputationFunction<<<1,1>>>();
    cudaDeviceSynchronize();
   

}

