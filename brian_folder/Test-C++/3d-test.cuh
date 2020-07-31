#include<iostream>

// Size of our matrix:
#define matrix_size 2
#define initial_value 1


class pencilComputation {
  public:

    int inputMatrix[matrix_size][matrix_size][matrix_size] = { { {initial_value} } };
    int outputMatrix[matrix_size][matrix_size][matrix_size] = { { { 0 } } };
    int pencilVector[matrix_size * matrix_size * matrix_size] = { 0 };

    void launcher();

    void CStats(const char *message);
    void DStats(const char *message);
};

