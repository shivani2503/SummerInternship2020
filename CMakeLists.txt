# NCAR Notes:
# This version of the CMakeLists.txt file uses file globs as opposed to a long, long list of files - this makes things easier, but
# does require re-running cmake manually when files are added to the source tree.  That doesn't seem problematic here, so I'm taking
# this approach for now.  Seeing how the CU team works with the code would be useful.


cmake_minimum_required ( VERSION 3.0.2 )
project(samurai)


set ( MODE GPU )                                   # NCAR: Options are 'CPU' or 'GPU' - CPU enables OpenMP, GPU enables OpenACC (default: CPU)
set ( CMAKE_BUILD_TYPE Release )                   # Standard: Options are 'Debug' or 'Release', the latter is optimized (these ARE case-sensitive for now)

set (PETSC_COMPILE 0)				   #NCAR: compile with PETSC (1) or don't (0)	

set ( SOLVER_MAXITER 2000 )        # max iterations in the solver (src/CostFunction.cpp)
set ( SOLVER_INNER_MAXITER 1000 )  # max interations for inner loop with TN solver
set ( SOLVER_CG_EPSILON 1.0e-18 )  # epsilon for the Samurai NCG solver
set ( SOLVER_CG_BETA_TYPE 2 )      # specify beta for Samurai NCG solver (1 = PR  (orig), 2 = PRP+ (recommended), 3 = FR, 4 = DY, 5 = HZ)
set ( SOLVER_CG_CONV_TYPE 1)       # type of convergenve for Samurai CG (1 = ~step size (orig), 2 = ||g(X)||/||g(X0)||)
set ( SOLVER_SAMURAI 1)      	   # 1= truncated Newton solver, 2 = Samurai CG (orig), 3 = Petsc solver (TN by default)
      		     		   # Note: to use PETSC solver, must compile with petsc (note: Petsc solver type and params can be modified with run-tme options)    
set ( SOLVER_CONV_TOL 1.0e-4 ) 	   # solver convergence tolerance    
set ( SOLVER_INNER_CONV_TOL 1.0e-4)  # solver convergence tolerance for the inner loop with TN (usually = conv_tol for best results)
set ( SOLVER_VERBOSE 1)            # 1 = print out extra solver converegnce info. (0 =  no extra printing)

# ----- Users shouldn't need to change anything below this, hopefully ----- 


# Code standards:
set ( CMAKE_CXX_STANDARD 11 )     # C++11


# Add our own CMake Modules to the search path:
set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" )
set ( CMAKE_PREFIX_PATH ${CMAKE_MODULE_PATH} )

# List of packages:
if (MODE STREQUAL "GPU")
  find_package ( OpenACC REQUIRED )                           # For GPU versions
  set ( OpenACC_ACCEL_TARGET "tesla:cc70,managed,lineinfo" )  # NCAR: Target options for GPU version - unused in CPU version
  set ( ADDITIONAL_FLAGS "-Minfo=accel" )                     # NCAR: If you want to add custom C++ flags, add them here
else()
  find_package ( OpenMP REQUIRED )              # For CPU versions (required; run with 1 thread if we want serial times)
endif()
find_package ( NetCDF COMPONENTS CXX REQUIRED ) # I/O (needs the C++ bindings)
find_package ( HDF5 COMPONENTS CXX REQUIRED )   # I/O (needs the C++ bindings) 
find_package ( ZLIB REQUIRED )                  # Compressed I/O
find_package ( FFTW REQUIRED )                  # Calculations
find_package ( GeographicLib REQUIRED )         # Observation stuff
if (PETSC_COMPILE)
   find_package ( PETSc )
endif()
# If PETSc is found, we'll add a macro:
if ( PETSC_FOUND )
  add_definitions( -DPETSC -DPETSC -DPETSC_SKIP_COMPLEX)
endif()

# Configure the solver.inc file (settings for the minimize() solve in src/CostFunction.cpp)
configure_file ( include/solver.inc.in include/solver.inc )

# Set up where the executable goes:
set ( EXECUTABLE_OUTPUT_PATH build/release/bin )


# Globs of file names:
file ( GLOB KDTree_Sources_Glob src/kd/[a-zA-Z0-9]*.cc )
file ( GLOB RADX_Sources_Glob   src/Radx/**/[a-zA-Z0-9]*.cc )
file ( GLOB NCXX_Sources_Glob   src/Ncxx/**/[a-zA-Z0-9]*.cc )
file ( GLOB Common_Sources_Glob src/[a-zA-Z0-9]*.cpp src/XML/[a-zA-Z0-9]*.cpp )
file ( GLOB GPTL_Sources_Glob   src/timing/[a-zA-Z0-9]*.c src/timing/[a-zA-Z0-9]*.inc )


# Use those globs to set up the old aliases:
set ( kd_SRCS       ${KDTree_Sources_Glob} ) 
set ( radx_SRCS     ${RADX_Sources_Glob} )
set ( ncxx_SRCS     ${NCXX_Sources_Glob} )
set ( common_SRCS   ${Common_Sources_Glob} ${kd_SRCS} ${radx_SRCS} ${ncxx_SRCS} src/date/tz.cpp src/datetime.cpp )
set ( samurai_SRCS	${common_SRCS} src/main.cpp )
set ( gptl_SRCS     ${GPTL_Sources_Glob} )
      

# Add our additional flags:
add_compile_options( ${ADDITIONAL_FLAGS} )


# Add the OpenMP / OpenACC flags to certain files:
set_source_files_properties ( src/VarDriver3D.cpp src/CostFunction3D.cu src/BkgdObsLoaders.cpp src/RecursiveFilter.cpp PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS}" )

# Set up the GPTL macros:
set_source_files_properties ( ${gptl_SRCS} PROPERTIES COMPILE_FLAGS "-DTHREADED_OMP -DHAVE_NANOTIME " )

# Executable:
add_executable        ( samurai ${samurai_SRCS} ${gptl_SRCS} )

# NCAR: Unfortunately the PGI compiler doesn't work out of the box - these are to fix various issues (maybe there's a better way to do this?):
# Note that because of how PGI treats command-line arguments, this is still imperfect - eg, adding '-O1' to a release build doesn't completely
# override the earlier, default '-O3' - rather, the '-O3' sets various things on (like vectorization), that are NOT turned off by the subsequent
# '-O1'.  This is annoying, and I don't currently have a workaround.
if (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set ( CMAKE_CXX_FLAGS "" )
    set ( CMAKE_CXX_FLAGS_RELEASE "" )
    set_source_files_properties ( ${samurai_SRCS} PROPERTIES COMPILE_FLAGS "-O3 -fast" )
    if (MODE STREQUAL "GPU")
      set_source_files_properties ( src/CostFunction3D.cu src/BkgdObsLoaders.cpp src/RecursiveFilter.cpp PROPERTIES COMPILE_FLAGS "-O3 -fast ${OpenACC_CXX_FLAGS} -ta=${OpenACC_ACCEL_TARGET}" )
      target_link_libraries ( samurai -ta=${OpenACC_ACCEL_TARGET} )
    else()
       set_source_files_properties ( src/CostFunction3D.cu src/BkgdObsLoaders.cpp src/RecursiveFilter.cpp PROPERTIES COMPILE_FLAGS "-O3 -fast ${OpenMP_CXX_FLAGS}" )
    endif()
    set_source_files_properties ( src/BSplineD.cpp PROPERTIES COMPILE_FLAGS "-O1" ) # BSpineD.cpp has issues above -O1 with PGI 19.3 & 19.9 at least
  endif()
  #add_compile_options( -pedantic )
else() # Intel, GNU:
  add_compile_options( -Wall )
  set_source_files_properties ( src/CostFunction3D.cu src/BkgdObsLoaders.cpp src/RecursiveFilter.cpp PROPERTIES COMPILE_FLAGS "-O3 -fast ${OpenMP_CXX_FLAGS}" )
endif()


# Set up the include and link paths:
include_directories ( ${FFTW_INCLUDES} ${NETCDF_INCLUDES} include ${GeographicLib_INCLUDE_DIRS} ${PETSC_INC} )
link_directories    ( ${GeographicLib_LIBRARY_DIRS} )


#target_link_libraries ( samurai ${OpenACC_CXX_FLAGS} ${HDF5_LIBRARIES} ${NETCDF_LIBRARIES} ${FFTW_LIBRARIES} ${GeographicLib_LIBRARIES} ${NETCDF_CPP_LIBRARY} ${OpenMP_CXX_LIBRARIES} ${PETSC_LIB} bz2 z -ta=${OpenACC_ACCEL_TARGET} )
target_link_libraries ( samurai ${OpenACC_CXX_FLAGS} ${HDF5_LIBRARIES} ${NETCDF_LIBRARIES} ${FFTW_LIBRARIES} ${GeographicLib_LIBRARIES} ${NETCDF_CPP_LIBRARY} ${OpenMP_CXX_LIBRARIES} ${PETSC_LIB} bz2 z )
install               ( TARGETS samurai DESTINATION bin )


# Basic diagnostic output:
#message( "<< INFO: ${CMAKE_BUILD_TYPE} build for ${MODE} >>" )
message( "<< INFO: ${CMAKE_BUILD_TYPE} build for ${MODE} - CG using ${SOLVER_MAXITER} iterations and epsilon of ${SOLVER_CONV_TOL} >>")
