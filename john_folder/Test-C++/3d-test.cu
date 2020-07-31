#include "3d-test.cuh"
#include<iostream>
#include<stdio.h>

__host__ __device__
 void scalingFunction(int array[]) {
        

    for(int i=0; i<8; i++) {
        array[i] = array[i] * 2;
    }


}

__host__ __device__
void  distributeFunction(int array[],int x,int y){
      pencilComputation p2; 
 
      for(int z=0; z<8; z++) {
        p2.outputMatrix[x][y][z] = p2.pencilVector[z];
      }

}



__global__ 
void pencilComputationFunction(){
 

pencilComputation p1;
 

int i = threadIdx.x ;//+ blockIdx.x * blockDim.x;
   int j = threadIdx.y ; // + blockIdx.y * blockDim.y;
    int z = threadIdx.z;
   //for(int k=0; k<8 ; k++) {
     
    //for(int x=0;x <2; x++) {
      //for(int y=0; y<2; y++) {
        //  for(int l=0; l<2; l++) {

             printf("%d\t",p1.inputMatrix[i][j][z]);
          //}
          printf("\n");
      //}
//      printf("\n********************************************\n"); // }

      
    // p1.pencilVector[k] = p1.inputMatrix[i][j][k];
  // }
  // scalingFunction(p1.pencilVector);
  // distributeFunction(p1.pencilVector,i,j);
// printf("**************************************\n");

}

void pencilComputation::launcher(){
    
    cudaMallocManaged((void**)&inputMatrix,8*sizeof(int));
   

    for(int i=0; i<2; i++) {
      for(int j=0; j<2; j++) {
          for(int k=0; k<2; k++) {
              inputMatrix[i][j][k] = 10;
          }
      }
  }

  for(int i=0; i<2; i++) {
      for(int j=0; j<2; j++) {
          for(int k=0; k<2; k++) {
             printf("%d\t",inputMatrix[i][j][k]);
          }
          printf("\n");
      }
      printf("\n*********************cu**********************************************\n");}


    dim3 grid(1,1,1);
    dim3 block(2,2,2);

   // #pragma acc enter data copyin(inputMatrix) copyout(outputMatrix)
   pencilComputationFunction<<<grid,block>>>();
   //pencilComputationFunction<<<1,3>>>();
    cudaDeviceSynchronize();
   

}

