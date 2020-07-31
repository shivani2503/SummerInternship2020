#!/bin/bash

module purge
module load ncarenv/1.3
module load intel/18.0.5
#module load pgi/19.9
module load mpt/2.19
#module load openmpi
module load netcdf
module load fftw
module load ncarcompilers
module load cmake

export MODULEPATH=${MODULEPATH}:/glade/u/home/bdobbins/Software/Modules

module load PETSc

HERE="$( cd "$(dirname "$0")" ; pwd -P )"
export OMP_NUM_THREADS=36

#mpiexec_mpt -n 1  omplace $HERE/../build/release/bin/samurai $* -tao_monitor -tao_gttol 1e-4 -tao_bnk_ksp_rtol 1e-4
#mpiexec_mpt -n 1  omplace $HERE/../build/release/bin/samurai $* -tao_monitor -tao_gttol 1e-5 -tao_bnk_ksp_rtol 1e-5 -tao_bnk_ksp_max_it 10 -tao_max_it 1 -tao_bnk_ksp_monitor

#mpiexec_mpt -n 1  omplace $HERE/../build/release/bin/samurai $* -tao_monitor -tao_gttol 1e-5 -tao_bnk_ksp_rtol 1e-5 -tao_max_it 1 -tao_bnk_ksp_monitor
mpiexec_mpt -n 1  omplace $HERE/../build/release/bin/samurai $* -tao_monitor -tao_gttol 1e-4 -tao_bnk_ksp_rtol 1e-4
#mpiexec_mpt -n 1  omplace $HERE/../build/release/bin/samurai $* -tao_monitor -tao_gttol 1e-3 -tao_bnk_ksp_rtol 1e-3 -tao_bnk_ksp_monitor
#srun --ntasks=1 $HERE/../build/release/bin/samurai $* -tao_monitor -tao_gttol 1e-4 -tao_bnk_ksp_rtol 1e-4 -tao_bnk_ksp_monitor
#mpiexec_mpt -n 1  $HERE/../build/release/bin/samurai $*
