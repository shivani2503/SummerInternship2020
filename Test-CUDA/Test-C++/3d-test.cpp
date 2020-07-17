#include"3d-test.h"


void pencilComputation::scalingFunction(int array[],int x) {
  
    for(int i=0; i<10; i++) {
        array[i] = array[i] * 2 * x;
    }

}

void pencilComputation::distributeFunction(int array[],int x,int y){
  
      for(int z=0; z<10; z++) {
          
          outputMatrix[x][y][z] = pencilVector[z];
      }

}

void pencilComputation::pencilComputationFunction(){

    wrapper w1;
    w1.launcher();
}

