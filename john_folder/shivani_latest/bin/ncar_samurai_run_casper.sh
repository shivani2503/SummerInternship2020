#!/bin/bash





module purge
module load ncarenv/1.2
#module load gnu/9.1.0
#module load intel/19.0.2
module load pgi/19.9
#module load mpt/2.19
module load netcdf
module load ncarcompilers/0.5.0
module load cmake

module use /glade/work/cponder/SHARE/Modules/Latest
module use /glade/work/cponder/SHARE/Modules/Legacy

module use --append /glade/work/cponder/SHARE/Modules/Bundles

for dir in /glade/work/cponder/SHARE/Modules/PrgEnv/*/*
do
    module use --append $dir
done
module load cuda/10.2
#module load pgi/19.9
module load PrgEnv/PGI+OpenMPI/2019-04-30
module load fftw/3.3.8

HERE="$( cd "$(dirname "$0")" ; pwd -P )"
export OMP_NUM_THREADS=1

#nvvp $HERE/../build/release/bin/samurai $*
nvvp $HERE/../build/release/bin/samurai $*
