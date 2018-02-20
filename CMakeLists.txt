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

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
INCLUDE(my_macros)

OPTION(WITH_CUDA "Compile with CUDA support" ON)
OPTION(WITH_GLOG "Enable Google Log support" ON)
OPTION(WITH_IPP "Compile with Intel IPP support" OFF)
OPTION(WITH_IWYU "Compile with import-what-you-use information" OFF)
OPTION(WITH_PYTORCH "Compile PyTorch wrapper" ON)
OPTION(WITH_TESTS "Compile tests" ON)
OPTION(WITH_TORCH "Compile Torch wrapper" ON)

SET(CUDA_ARCH_LIST "Auto" CACHE STRING "Compile with support for these CUDA architectures")
SET(PYTHON_SETUP_PREFIX "" CACHE STRING "Prefix path for the Python setup scripts")

# C++ compiler options
ADD_COMPILE_OPTIONS(-std=c++11 -Wall -pedantic -fPIC)

INCLUDE(GetGitRevisionDescription)
GET_GIT_HEAD_REVISION(GIT_REFSPEC GIT_SHA1)
GIT_LOCAL_CHANGES(GIT_CLEAN)
GIT_GET_EXACT_TAG(GIT_TAG)
STRING(REGEX REPLACE ".*/([^/]+)" "\\1" GIT_BRANCH "${GIT_REFSPEC}")
STRING(SUBSTRING "${GIT_SHA1}" 0 7 GIT_SHA1_SHORT)

SET(IMGDISTORT_VERSION 0.1.0)
IF(GIT_CLEAN STREQUAL "DIRTY")
  SET(IMGDISTORT_VERSION "${IMGDISTORT_VERSION}+${GIT_SHA1_SHORT}.dirty")
ELSEIF(GIT_TAG)
  SET(IMGDISTORT_VERSION "${GIT_TAG}")
ELSE()
  SET(IMGDISTORT_VERSION "${IMGDISTORT_VERSION}+${GIT_SHA1_SHORT}")
ENDIF()

FIND_PACKAGE(CUDA)
FIND_PACKAGE(Glog)
FIND_PACKAGE(GMock)
FIND_PACKAGE(GTest)
FIND_PACKAGE(IPP)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(PyTorch)
FIND_PACKAGE(Torch)

# CUDA compiler flags
IF(CUDA_FOUND AND WITH_CUDA)
  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
  LIST(APPEND CUDA_NVCC_FLAGS -std=c++11)
  LIST(APPEND CUDA_NVCC_FLAGS -lineinfo)
  IF(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5)
    LIST(APPEND CUDA_NVCC_FLAGS -D_MWAITXINTRIN_H_INCLUDED)
    LIST(APPEND CUDA_NVCC_FLAGS -D_FORCE_INLINES)
  ENDIF()
  CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA ${CUDA_ARCH_LIST})
  LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})

  # Propagate C++ compiler options
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -Wall,-fPIC")
  STRING(REGEX REPLACE "[ ]+" "," _cxx_host_extra_flags
    "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler ${_cxx_host_extra_flags}")
ELSE()
  SET(WITH_CUDA OFF)
ENDIF(CUDA_FOUND AND WITH_CUDA)

# Glog flags
IF(GLOG_FOUND AND WITH_GLOG)
  INCLUDE_DIRECTORIES("${GLOG_INCLUDE_DIRS}")
  LINK_LIBRARIES("${GLOG_LIBRARIES}")
ELSE()
  SET(WITH_GLOG OFF)
ENDIF(GLOG_FOUND AND WITH_GLOG)

# Google Test / Mock flags
IF(GMOCK_FOUND AND GTEST_FOUND AND WITH_TESTS)
ELSE()
  SET(WITH_TESTS OFF)
ENDIF(GMOCK_FOUND AND GTEST_FOUND AND WITH_TESTS)

# IPP flags
IF(IPP_FOUND AND WITH_IPP)
  IF(IPP_VERSION_MAJOR VERSION_GREATER 8)
    MESSAGE(STATUS "This IPP version is not supported!")
    SET(WITH_IPP OFF)
  ELSE()
    INCLUDE_DIRECTORIES("${IPP_INCLUDE_DIRS}")
  ENDIF()
ELSE()
  SET(WITH_IPP OFF)
ENDIF()

# OpenMP flags
IF(OPENMP_FOUND)
  ADD_COMPILE_OPTIONS(${OpenMP_CXX_FLAGS})
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

# PyTorch flags
IF(PYTORCH_FOUND AND WITH_PYTORCH)
  IF(PYTORCH_CUDA AND NOT WITH_CUDA)
    MESSAGE(WARNING "Your PyTorch distribution supports CUDA, but "
      "CUDA support for imgdistort is disabled!")
    SET(PYTORCH_CUDA OFF)
  ENDIF()
ELSE()
  SET(WITH_PYTORCH OFF)
ENDIF()

# Torch flags
IF(TORCH_FOUND AND WITH_TORCH)
  IF(TORCH_CUDA AND NOT WITH_CUDA)
    MESSAGE(WARNING "Your Torch distribution supports CUDA, but "
      "CUDA support for imgdistort is disabled!")
    SET(TORCH_CUDA OFF)
  ENDIF()
ELSE()
  SET(WITH_TORCH OFF)
ENDIF()

# -----------------------------------------------------------------------------
# C++ Library
# -----------------------------------------------------------------------------
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}")


MESSAGE(STATUS "WITH_CUDA = ${WITH_CUDA}")
MESSAGE(STATUS "WITH_GLOG = ${WITH_GLOG}")
MESSAGE(STATUS "WITH_IPP = ${WITH_IPP}")
MESSAGE(STATUS "WITH_IWYU = ${WITH_IWYU}")
MESSAGE(STATUS "WITH_PYTORCH = ${WITH_PYTORCH}")
MESSAGE(STATUS "WITH_TESTS = ${WITH_TESTS}")
MESSAGE(STATUS "WITH_TORCH = ${WITH_TORCH}")

ADD_SUBDIRECTORY(imgdistort)
ADD_SUBDIRECTORY(THW)
ADD_SUBDIRECTORY(pytorch)
#ADD_SUBDIRECTORY(torch)
