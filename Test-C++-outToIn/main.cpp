#include"3d-test.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>

int main() {

  // Size of our n x n x n matrix:
  int n = matrix_size;

  // Initial value of each entry in our matrix:
  //int initial_value = 1;

  //Size(in bytes) of matrix
  //size_t bytes = n * n * sizeof(int);

  //Host pointers
  //int inputMatrix[n][n][n];
  
  //Allocate host memory
  //cudaMallocManaged(inputMatrix[][][],bytes);
  pencilComputation obj1;
   
  /*Initialize input_Matrix on CPU */
  for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {
      for(int k=0; k<n; k++) {
        obj1.inputMatrix[i][j][k] = initial_value;
      }
    }
  }
  obj1.CStats("Post-Initialization"); 
  // Object to access kernel function on GPU
  //#pragma acc data copy(obj1) 
  pencilComputation *cuda_obj1;
  cudaMalloc((void**)&cuda_obj1, sizeof(pencilComputation));
  printf("[NOTE] Allocated %d bytes for cuda_obj1\n\n", sizeof(pencilComputation));
  cudaMemcpy(cuda_obj1, &obj1, sizeof(pencilComputation), cudaMemcpyHostToDevice);
  {
     cuda_obj1->launcher();
  }
  cudaMemcpy(&obj1, cuda_obj1, sizeof(pencilComputation), cudaMemcpyDeviceToHost);
  obj1.CStats("After Kernel call"); 


  // Print results on CPU 
std::cout<<"outputMatrix Elements\n";  
for(int i=0; i<2; i++) {
      for(int j=0; j<2; j++) {
          for(int k=0; k<2; k++) {
             std::cout<<obj1.outputMatrix[i][j][k]<<"\t";
          }
          std::cout<<"\n";
      }
      std::cout<<"\n***********************************************************************************************\n";
  }

return 0;

}
