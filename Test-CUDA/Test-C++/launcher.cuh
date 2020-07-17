#include<stdio.h>
#include "3d-test.h"

#include "device_launch_parameters.h"

class wrapper{

public:
    void launcher();
    __global__ void pencilComputationSubPart();

};
