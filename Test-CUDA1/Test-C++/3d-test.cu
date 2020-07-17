#include"3d-test.h"
#include"launcher.cuh"

void pencilComputation::scalingFunction(int array[],int x) {
  
    for(int i=0; i<10; i++) {
        array[i] = array[i] * 2 * x;
    }

}

void pencilComputation::distributeFunction(int array[],int x,int y){
  
      for(int z=0; z<10; z++) {
          
          outputMatrix[x][y][z] = pencilVector[z];
      }

}

__global__ void pencilComputation::pencilComputationFunction(){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   int k = threadIdx.z + blockIdx.z * blockDim.z;
   scalingFunction(obj1.pencilVector,i);
   distributeFunction(obj1.pencilVector,i,j);

}

pencilComputation::launcher(){
    
    dim3 grid(1,1,1);
    dim3 block(10,10,10);

    pencilComputationSubPart<<<grid,block>>>();
}

