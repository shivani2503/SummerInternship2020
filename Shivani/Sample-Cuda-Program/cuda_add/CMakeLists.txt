cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

#project(cmake-cuda-add )

#enable_language(CUDA)

#add the executable
cuda_add_executable(cmake-cuda-add cuda-add.cu cuda-add-main.cu)
