#include <math.h>
#include "cuda-add.h"

// function to add the elements of two arrays

void add(int n, float *x, float *y)
 {
   for (int i = 0; i < n; i++)
         y[i] = x[i] + y[i];
 }


