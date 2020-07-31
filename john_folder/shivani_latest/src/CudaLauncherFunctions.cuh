#include<iostream>
#include"CostFunction3D.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

class launcherFunctions {
public:

void launcherFunction_SCtransform(const real* Astate, real* Cstate,int var, CostFunction3D& obj1, int order, double *alpha, double beta, double [5][5]);


};

bool CUDAfilterArray(double* array, double *q, double *s, const int& arrLength, const int &order, double *alpha, double beta, double [5][5]);
