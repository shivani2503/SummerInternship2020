#include"CostFunction3D.h"
#include"CudaLauncherFunctions.cuh"
#include<iostream>

#define INDEX(i, j, k, idim, jdim, vdim, var) ((vdim) * ((idim) * ((jdim) * (k) + j) + i) + var)

__device__
void Sum(real* state)
{
  real sum = 0.0;
  //printf("Sum: %g\n", sum);
  for (int i = 0; i < 334425; i++) {
  //for (int i = 0; i < 3; i++) {
    sum += state[i];
  }
  //printf("Sum: %g\n", sum);
}

//__host__ __device__
__device__
void gatherFunctionK(int i,int j,const real* Astate,int var,real *kTemp, int iDim, int jDim, int kDim, int varDim ) {
  //printf("Inside gatherFunctionK : %d, %d, %d, %d, %d, %d, %d\n", i, j, var, iDim, jDim, kDim, varDim);
  
  int index;  

  for(int k=0;k<kDim;k++) {

     index = INDEX(i,j,k,iDim,jDim,varDim,var);  
     kTemp[k] = Astate[index];
  }
  //printf("Ending gatherFunctionK : %d, %d, %d, %d, %d, %d, %d\n", i, j, var, iDim, jDim, kDim, varDim);

 }


__host__ __device__
void scatterFunctionK(int i,int j,real* Cstate,int var,real *kTemp, int iDim, int jDim, int kDim, int varDim) {
  //printf("Inside scatterFunctionK : %d, %d, %d, %d, %d, %d, %d\n", i, j, var, iDim, jDim, kDim, varDim);

  int index;

  for(int k=0;k<kDim;k++) {

     index = INDEX(i,j,k,iDim,jDim,varDim,var);
     Cstate[index] = kTemp[k];
  }

}

__global__
void pencilComputation(const real* Astate, real* Cstate, int var, int iDim, int jDim, int kDim, int varDim) {
//void pencilComputation(const real* Astate, real* Cstate, int var,CostFunction3D obj4) {
  
//printf("DEBUG2: Inside pencilComputation GPU call\n"); 
//printf("DEBUG2: Astate[0], Cstate[0], var, iDim, jDim, kDim, varDim\n");
//printf("DEBUG2: %g, %g, %d, %d, %d, %d, %d\n", Astate[0], Cstate[0], var, iDim, jDim, kDim, varDim);
 // int index;
  real* kTemp = new real[kDim];
  //real* kTemp;
  //cudaMalloc((void **)&kTemp, sizeof(real) * kDim);
  //real kTemp[39];
  //real *kq = new real[kDim];
  //real *ks = new real[kDim];
//printf("DEBUG: step 2\n"); 

  //int i = threadIdx.x;
  //int j = threadIdx.y;
  //int bi = blockIdx.x;
  //int bj = blockIdx.y;
  //printf("DEBUG: step 3: %d %d %d %d \n",i,j,bi,bj); 
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("DEBUG: step 3: %d %d: %d %d \n",blockIdx.x,blockIdx.y,i,j); 
  //Sum(Cstate);
  //printf("DEBUG: step 4\n"); 
  gatherFunctionK(i,j,Astate,var,kTemp, iDim, jDim, kDim, varDim);
  //printf("DEBUG: step 5\n"); 
  
  //Sum(Cstate);
  //if (obj4.kFilterScale > 0) obj4.kFilter->filterArray(kTemp, kq,ks,obj4.kDim);

  scatterFunctionK(i,j,Cstate,var,kTemp, iDim, jDim, kDim, varDim);
  //Sum(Cstate);
  delete [] kTemp;

}


//#pragma acc routine seq
void launcherFunctions::launcherFunction_SCtransform(const real* Astate, real* Cstate,int var,CostFunction3D& obj1 ) 
{
  //Function call for Line#1389 - Line#1391
 dim3 grid(obj1.iDim,obj1.jDim,obj1.kDim);
 //dim3 block(obj1.iDim,obj1.jDim,obj1.kDim);
 dim3 block(1,1,1);
 
 //printf("Astate pointer (unknown) : %p\n", Astate);
 //printf("Cstate pointer (unknown) : %p\n", Cstate);
//printf("DEBUG1: Before pencilComputation GPU call\n"); 
//printf("DEBUG1: Astate[0], Cstate[0], var, iDim, jDim, kDim, varDim\n");
//printf("DEBUG1: %g, %g, %d, %d, %d, %d, %d\n", Astate[0], Cstate[0], var, obj1.iDim, obj1.jDim, obj1.kDim, obj1.varDim);
//pencilComputation<<<grid,block>>>(Astate,Cstate,var,obj1.iDim, obj1.jDim, obj1.kDim, obj1.varDim); 
pencilComputation<<<grid,block>>>(Astate,Cstate,var,obj1.iDim, obj1.jDim, obj1.kDim, obj1.varDim); 
//printf("DEBUG: After pencilComputation GPU call\n"); 
cudaDeviceSynchronize();

}
