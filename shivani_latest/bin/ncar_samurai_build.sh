#!/bin/bash

module purge
module load ncarenv/1.2
#module load gnu/9.1.0
module load intel/19.0.2
#module load pgi/19.9
module load mpt/2.19
module load netcdf/4.6.3
module load fftw/3.3.8
module load ncarcompilers/0.5.0
module load cmake

HERE="$( cd "$(dirname "$0")" ; pwd -P )"
cd $HERE/..

rm -rf CMakeFiles/
rm CMakeCache.txt

cmake .

make -j 4
