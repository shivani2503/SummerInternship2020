__global__
void multiply(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] * y[i];
}
