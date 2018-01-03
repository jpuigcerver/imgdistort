CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(imgdistortion CXX)
ENABLE_TESTING()

# The project should not be compiled from the source directory, since the
# compilation and building process will generate files and some of them may
# overwrite original source files. It is better to avoid that and ask the user
# to compile the project from a separate folder (i.e. a build directory).
IF(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  MESSAGE(FATAL_ERROR
    "In-source builds not allowed. "
    "Please make a build directory and run CMake from there "
    "(you may need to remove CMakeCache.txt and CMakeFiles).")
ENDIF()

OPTION(WITH_CUDA "Compile with CUDA support" ON)
OPTION(WITH_IPP "Compile with Intel IPP support" ON)
OPTION(WITH_TESTS "Compile tests" ON)
OPTION(WITH_IWYU "Compile with import-what-you-use information" OFF)
OPTION(WITH_PYTORCH "Compile PyTorch wrapper" ON)
OPTION(WITH_TORCH "Compile Torch wrapper" ON)

SET(CUDA_ARCH_LIST "Auto" CACHE STRING "Compile with support for these CUDA architectures")
SET (PYTHON_SETUP_PREFIX "" CACHE STRING "Prefix path for the Python setup scripts.")


# C++ compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_FORCE_INLINES -Wall -pedantic -fPIC")

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
FIND_PACKAGE(Glog REQUIRED)
FIND_PACKAGE(CUDA)
FIND_PACKAGE(IPP)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(PyTorch)
FIND_PACKAGE(Torch)


INCLUDE_DIRECTORIES("${GLOG_INCLUDE_DIRS}")
# IPP flags
IF(IPP_FOUND AND WITH_IPP)
  INCLUDE_DIRECTORIES("${IPP_INCLUDE_DIRS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_IPP")
ELSE()
  SET(WITH_IPP OFF)
ENDIF()

# OpenMP flags
IF(OPENMP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

# CUDA compiler flags
IF(CUDA_FOUND AND WITH_CUDA)
  IF(NOT COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/select_compute_arch.cmake)
  ENDIF()

  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_CUDA")
  SET(CUDA_NVCC_FLAGS "-std=c++11 -lineinfo -Xcompiler -Wall,${OpenMP_CXX_FLAGS},-fPIC")
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DWITH_CUDA")
  IF(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5)
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES")
  ENDIF()
  CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA ${CUDA_ARCH_LIST})
  LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
ELSE()
  SET(WITH_CUDA OFF)
ENDIF(CUDA_FOUND AND WITH_CUDA)


FIND_PROGRAM(IWYU_PATH NAMES include-what-you-use iwyu)
MACRO(ADD_IWYU_TO_TARGET)
  IF(WITH_IWYU AND IWYU_PATH AND CMAKE_VERSION VERSION_GREATER "3.2")
    FOREACH(_a ${ARGN})
      SET_PROPERTY(TARGET ${_a} PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${IWYU_PATH})
    ENDFOREACH()
  ENDIF()
ENDMACRO(ADD_IWYU_TO_TARGET)

# -----------------------------------------------------------------------------
# C++ Library
# -----------------------------------------------------------------------------
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/include")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")

ADD_SUBDIRECTORY(include/imgdistort)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(pytorch)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(torch)
