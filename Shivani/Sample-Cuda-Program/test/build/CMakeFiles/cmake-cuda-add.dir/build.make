# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build

# Include any dependencies generated for this target.
include CMakeFiles/cmake-cuda-add.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cmake-cuda-add.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cmake-cuda-add.dir/flags.make

CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add.cu.o: CMakeFiles/cmake-cuda-add.dir/cmake-cuda-add_generated_cuda-add.cu.o.depend
CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add.cu.o: CMakeFiles/cmake-cuda-add.dir/cmake-cuda-add_generated_cuda-add.cu.o.cmake
CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add.cu.o: ../cuda-add.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object CMakeFiles/cmake-cuda-add.dir//./cmake-cuda-add_generated_cuda-add.cu.o"
	cd /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir && /usr/bin/cmake -E make_directory /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir//.
	cd /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir//./cmake-cuda-add_generated_cuda-add.cu.o -D generated_cubin_file:STRING=/glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir//./cmake-cuda-add_generated_cuda-add.cu.o.cubin.txt -P /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir//cmake-cuda-add_generated_cuda-add.cu.o.cmake

CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add-main.cu.o: CMakeFiles/cmake-cuda-add.dir/cmake-cuda-add_generated_cuda-add-main.cu.o.depend
CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add-main.cu.o: CMakeFiles/cmake-cuda-add.dir/cmake-cuda-add_generated_cuda-add-main.cu.o.cmake
CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add-main.cu.o: ../cuda-add-main.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object CMakeFiles/cmake-cuda-add.dir//./cmake-cuda-add_generated_cuda-add-main.cu.o"
	cd /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir && /usr/bin/cmake -E make_directory /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir//.
	cd /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir//./cmake-cuda-add_generated_cuda-add-main.cu.o -D generated_cubin_file:STRING=/glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir//./cmake-cuda-add_generated_cuda-add-main.cu.o.cubin.txt -P /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir//cmake-cuda-add_generated_cuda-add-main.cu.o.cmake

# Object files for target cmake-cuda-add
cmake__cuda__add_OBJECTS =

# External object files for target cmake-cuda-add
cmake__cuda__add_EXTERNAL_OBJECTS = \
"/glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add.cu.o" \
"/glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add-main.cu.o"

cmake-cuda-add: CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add.cu.o
cmake-cuda-add: CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add-main.cu.o
cmake-cuda-add: CMakeFiles/cmake-cuda-add.dir/build.make
cmake-cuda-add: /glade/u/apps/dav/opt/cuda/10.1/lib64/libcudart.so
cmake-cuda-add: CMakeFiles/cmake-cuda-add.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable cmake-cuda-add"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cmake-cuda-add.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cmake-cuda-add.dir/build: cmake-cuda-add
.PHONY : CMakeFiles/cmake-cuda-add.dir/build

CMakeFiles/cmake-cuda-add.dir/requires:
.PHONY : CMakeFiles/cmake-cuda-add.dir/requires

CMakeFiles/cmake-cuda-add.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cmake-cuda-add.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cmake-cuda-add.dir/clean

CMakeFiles/cmake-cuda-add.dir/depend: CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add.cu.o
CMakeFiles/cmake-cuda-add.dir/depend: CMakeFiles/cmake-cuda-add.dir/./cmake-cuda-add_generated_cuda-add-main.cu.o
	cd /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build /glade/u/home/shivanis/Shivani/Sample-Cuda-Program/cuda_add/build/CMakeFiles/cmake-cuda-add.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cmake-cuda-add.dir/depend
