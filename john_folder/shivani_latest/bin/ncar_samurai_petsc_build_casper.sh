#!/bin/bash

module purge
module load ncarenv/1.3
#module load intel
module load pgi/19.9
module load openmpi
module load netcdf
module load fftw
module load ncarcompilers
module load cmake

export MODULEPATH=${MODULEPATH}:/glade/u/home/bdobbins/Software/Modules
module load PETSc

HERE="$( cd "$(dirname "$0")" ; pwd -P )"
cd $HERE/..

rm -rf CMakeFiles/
rm CMakeCache.txt

CXX=mpicxx cmake .

make -j 4
