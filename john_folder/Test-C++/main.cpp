#include"3d-test.cuh"

int main() {

   //Matrix size of 2 X 2 X 2 
   //int n = 1 << 1;

   //Size(in bytes) of matrix
   //size_t bytes = n * n * sizeof(int);

      //Host pointers
   //int inputMatrix[n][n][n];
  
   //Allocate host memory
   //cudaMallocManaged(inputMatrix[][][],bytes);
  pencilComputation obj1;
   
  /*Initialize input_Matrix on CPU */
    for(int i=0; i<2; i++) {
      for(int j=0; j<2; j++) {
          for(int k=0; k<2; k++) {
             obj1. inputMatrix[i][j][k] = 10;
          }
      }
  }

for(int i=0; i<2; i++) {
      for(int j=0; j<2; j++) {
          for(int k=0; k<2; k++) {
             std::cout<<obj1.inputMatrix[i][j][k]<<"\t";
          }
          std::cout<<"\n";
      }
      std::cout<<"\n********************cpp***********************************************\n";
  }

  // Object to access kernel function on GPU
  #pragma acc data copy(obj1.inputMatrix) 
  {
     obj1.launcher();
  }

  // Print results on CPU 
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
