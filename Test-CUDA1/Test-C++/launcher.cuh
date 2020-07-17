#include<stdio.h>
#include"3d-test.h"
#include"cuda_runtime"
#include"device_launch_parameters.h"

class wrapper{

public:
    void launcher();
    __global__ void pencilComputationSubPart();

}