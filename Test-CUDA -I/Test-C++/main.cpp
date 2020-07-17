#include"3d-test.cuh"

int main() {

   pencilComputation obj1;

   for(int i=0; i<10; i++) {
      for(int j=0; j<10; j++) {
          for(int k=0; k<10; k++) {
              obj1.inputMatrix[i][j][k] = 10;
          }
      }
  }

  obj1.launcher();

  for(int i=0; i<10; i++) {
      for(int j=0; j<10; j++) {
          for(int k=0; k<10; k++) {
              
             std::cout<<obj1.outputMatrix[i][j][k]<<"\t";
          }
          std::cout<<"\n";
      }
      std::cout<<"\n***********************************************************************************************\n";
  }

}