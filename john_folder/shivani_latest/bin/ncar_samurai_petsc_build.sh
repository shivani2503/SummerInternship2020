#!/bin/bash

module purge
module load ncarenv/1.3
module load intel/18.0.5
#module load pgi/19.9

#MPT
module load mpt/2.19

#OpenMPI
#module load openmpi

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

echo "before srun"
CXX=mpicxx cmake .

make -j 4
