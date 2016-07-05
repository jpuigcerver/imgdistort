cmake_minimum_required(VERSION 3.0)
project(cuda_kernels C CXX)

# The project should not be compiled from the source directory, since the
# compilation and building process will generate files and some of them may
# overwrite original source files. It is better to avoid that and ask the user
# to compile the project from a separate folder (i.e. a build directory).
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR
    "In-source builds not allowed. "
    "Please make a build directory and run CMake from there "
    "(you may need to remove CMakeCache.txt and CMakeFiles).")
endif()


add_subdirectory(src)
