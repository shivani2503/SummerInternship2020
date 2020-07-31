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
void pencilComputation(const real* Astate, real* Cstate, int var, int iDim, int jDim, int kDim, int varDim, CostFunction3D &obj, int kFilterScale, int order, double *alpha, double beta, double Sn[5][5]) {
//void pencilComputation(const real* Astate, real* Cstate, int var,CostFunction3D obj4) {
  

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("gridsize: %d %d %d\n",iDim,jDim,kDim);
  //printf("threadIdx,blockDim,blockIdx: %d %d, %d %d, %d %d, %d %d \n",threadIdx.x,threadIdx.y,blockDim.x,blockDim.y,blockIdx.x,blockIdx.y,i,j);
  if (i < iDim && j < jDim) {
    real *kTemp, *kq, *ks;
    kTemp = (real *) malloc(sizeof(real) * kDim);
    kq    = (real *) malloc(sizeof(real) * kDim);
    ks    = (real *) malloc(sizeof(real) * kDim);

    gatherFunctionK(i,j,Astate,var,kTemp, iDim, jDim, kDim, varDim);

    if (kFilterScale > 0) {CUDAfilterArray(kTemp, kq, ks, kDim, order, alpha, beta, Sn);}

    scatterFunctionK(i,j,Cstate,var,kTemp, iDim, jDim, kDim, varDim);

    free(kTemp);
    free(kq);
    free(ks);
  }

}


#pragma acc seq
void launcherFunctions::launcherFunction_SCtransform(const real* Astate, real* Cstate,int var,CostFunction3D& obj1, int order, double *alpha, double beta, double Sn[5][5]) 
{
  int blkx = 4;
  //Function call for Line#1389 - Line#1391
  dim3 grid(obj1.iDim/blkx,obj1.jDim);
  dim3 block(blkx,1);

 
 //printf("Astate pointer (unknown) : %p\n", Astate);
 // printf("Cstate pointer (unknown) : %p\n", Cstate);
//printf("DEBUG1: Before pencilComputation GPU call\n"); 
//printf("DEBUG1: Astate[0], Cstate[0], var, iDim, jDim, kDim, varDim\n");
//printf("DEBUG1: %g, %g, %d, %d, %d, %d, %d\n", Astate[0], Cstate[0], var, obj1.iDim, obj1.jDim, obj1.kDim, obj1.varDim);
//pencilComputation<<<grid,block>>>(Astate,Cstate,var,obj1.iDim, obj1.jDim, obj1.kDim, obj1.varDim); 
pencilComputation<<<grid,block>>>(Astate,Cstate,var,obj1.iDim, obj1.jDim, obj1.kDim, obj1.varDim, obj1, obj1.kFilterScale, order, alpha, beta, Sn); 
cudaDeviceSynchronize();

}

__device__
double RecursiveFilter::CUDAfactorial(const double& max) 
{
	double n = 1;
	for (double i=2;i<=max;i++) {
		n *= i;
	}
	return n;
	
}

__device__
void CUDAsolveBC(double* A, double* B, double S[5][5], int order) 
{
	
	int n = order-1;
	for(int j=0;j<=n-1;j++) {
		for (int i=j+1;i<=n;i++) {
			S[i][j]=S[i][j]/S[j][j];
		}
		for	(int i=j+1;i<=n;i++) {
			for (int k=j+1;k<=n;k++) {
				S[i][k]=S[i][k]-S[i][j]*S[j][k];
			}
			B[i]=B[i]-S[i][j]*B[j];
		}
	}
	A[n]=B[n]/S[n][n];
	for(int j=n-1;j>=0;j--) {
		A[j]=B[j];
		for(int k=n;k>=j+1;k--) {
			A[j]=A[j]-A[k]*S[j][k];
		}
		A[j]=A[j]/S[j][j];
	}
}

__device__ 
bool CUDAfilterArray(double* array, double *q, double *s, const int& arrLength, const int &order, double *alpha, double beta, double Sn[5][5])
{ 
//  printf("CUDAfilterArray: arrLength = %d, %d, %p, %p\n", arrLength, order, q, s);
//        printf("CUDAfilterArray: array = %p\n", array);
//        printf("CUDAfilterArray: q     = %p\n", q    );
//        printf("CUDAfilterArray: s     = %p\n", s    );
	int maxi = arrLength-1;
	double* p = array;
	double A[4],B[4];

#if 0
        double q[arrLength],s[arrLength];
        //double A[4],B[4];
	//double* q = new double[arrLength];
	//double* s = new double[arrLength];
	//
 	double * q, * s;
	q = (double *) malloc(sizeof(double)*arrLength);
	s = (double *) malloc(sizeof(double)*arrLength);
#endif


	for (int i=0; i<= maxi; i++) {
		q[i] = 0;
		s[i] = 0;
	}
        
	
	q[0]=beta*p[0];
	q[1]=beta*p[1] + alpha[1]*q[0];
	q[2]=beta*p[2] + alpha[1]*q[1] + alpha[2]*q[0];
	q[3]=beta*p[3] + alpha[1]*q[2] + alpha[2]*q[1] + alpha[3]*q[0];	
	for (int i=order; i<= maxi; i++) {
		q[i] = beta*p[i] + alpha[1]*q[i-1]
		+ alpha[2]*q[i-2] + alpha[3]*q[i-3] + alpha[4]*q[i-4];
	}
	
    // Invert Sn
	// Create a temporary copy of Sn just in case it gets thrashed in the Gaussian Elimination
	double Stmp[5][5];
	for (int i=0;i<=4;i++) {
		for (int j=0;j<=4;j++) {
			Stmp[i][j] = Sn[i][j];
		}
	}

	//double* A = new double[4];
	//double* B = new double[4];
	for (int i=maxi-order+1; i<= maxi; i++) {
		B[i-(maxi-order+1)] = q[i];
	}
	CUDAsolveBC(A, B, Stmp, order);
	for (int i=maxi; i>= (maxi-order+1); i--) {
		s[i] = A[i-(maxi-order+1)];
	}
	for (int i=maxi-order;i>=0;i--) {
		s[i] = beta*q[i] + alpha[1]*s[i+1]
		+ alpha[2]*s[i+2] + alpha[3]*s[i+3] + alpha[4]*s[i+4];
		// std::cout << s[i] << std::endl;
	}
	//delete[] A;
	//delete[] B;
	
	for (int i=0; i<= maxi; i++) {
		// To get a 'true' Gaussian, need to scale by this factor
		// To preserve total quantity (needed for Variational analysis)
		// Do not scale resulting vector
		//double Pi = 3.141592653589793238462643;
		//array[i] = s[i]*sqrt(2*Pi)*lengthScale;
		array[i] = s[i];
	}
	//delete[] q;
	//delete[] s;
#if 0
        free(q);
        free(s);
#endif
	return true;
	
}



