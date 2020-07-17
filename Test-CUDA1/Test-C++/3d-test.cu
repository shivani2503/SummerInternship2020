#include "3d-test.cuh"

__host__ __device__
 void scalingFunction(int array[],int x) {
        

    for(int i=0; i<10; i++) {
        array[i] = array[i] * 2 * x;
    }


}

__host__ __device__
void  distributeFunction(int array[],int x,int y){
      pencilComputation p2; 
 
      for(int z=0; z<10; z++) {
          
          p2.outputMatrix[x][y][z] = p2.pencilVector[z];
      }

}

__global__ 
void pencilComputationFunction(){
 
pencilComputation p1;
   
int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   
   for(int k=0; k<10 ; k++) {
 
     p1.pencilVector[k] = p1.inputMatrix[i][j][k];

   }
    scalingFunction(p1.pencilVector,i);
    distributeFunction(p1.pencilVector,i,j);
 
}

void pencilComputation::launcher(){
    
    dim3 grid(1,1,1);
    dim3 block(10,10,10);

    pencilComputationFunction<<<grid,block>>>();
    cudaDeviceSynchronize();
   

}

