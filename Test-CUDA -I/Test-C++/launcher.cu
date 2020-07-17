#include "launcher.cuh"


wrapper::pencilComputationSubPart(){

   pencilComputation obj1;

   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   int k = threadIdx.z + blockIdx.z * blockDim.z;
   scalingFunction(obj1.pencilVector,i);
   distributeFunction(obj1.pencilVector,i,j);

}

wrapper::launcher() {

dim3 grid(1,1,1);
dim3 block(10,10,10);

pencilComputationSubPart<<<grid,block>>>();

}