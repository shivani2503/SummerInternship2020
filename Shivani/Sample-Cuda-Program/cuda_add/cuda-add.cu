#include<math.h>
#include "cuda-add.h"


cuda-add::void add(int n, float *x, float *y)
 {
   for (int i = 0; i < n; i++)
         y[i] = x[i] + y[i];
 }



