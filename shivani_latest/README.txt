ASAP version of Samurai, using just C++ (no Qt)

To run after downloading:

1) 
module purge
module load ncarenv/1.2
module load gnu/9.1.0
module load mpt/2.19
module load netcdf/4.6.3
module load fftw/3.3.8
module load ncarcompilers/0.5.0

2) cmake .

3) make -j 4


Note that the CMake modules are pretty messed up, so libGeographic is hardcoded for the moment, you need the 'ncarcompilers' module or it doesn't find the FFTW includes, etc.  
This is something to change in the near future.

Finally, while this runs with the single ob case and looks the same, running the supercell case shows some early differences - notably, the original code finds 4421542 observations, vs. 4372390 for this one.  I'm *guessing* this has to do with time comparisons, but I'll start to look into it.  Nevertheless, for analyzing performance it should be representative until I solve the remaining issues.

The code is pretty ugly, and start-up (reading observations) seems slower in my way - possibly because I've been building in debug mode, whereas Qt handles things in a release mode, but I'll also look into that soon.
