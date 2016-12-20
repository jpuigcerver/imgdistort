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

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
FIND_PACKAGE(Glog REQUIRED)
FIND_PACKAGE(CUDA)
FIND_PACKAGE(IPP)
FIND_PACKAGE(OpenMP)

# C++ compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_FORCE_INLINES")
# OpenMP flags
IF(OPENMP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()
# IPP flags
IF(WITH_IPP AND IPP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_IPP")
ENDIF()

# CUDA compiler flags
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std c++11 -Xcompiler -fPIC,-D_FORCE_INLINES")


# ------------------------------------------------------------------------------
# C++ Library
# ------------------------------------------------------------------------------
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/include")
INCLUDE_DIRECTORIES("${GLOG_INCLUDE_DIRS}")
#CUDA_ADD_LIBRARY(affine affine.cu affine.h utils.h STATIC)
#TARGET_LINK_LIBRARIES(affine ${GLOG_LIBRARIES})

#CUDA_ADD_LIBRARY(morphology morphology.cu morphology.h utils.h STATIC)
#TARGET_LINK_LIBRARIES(morphology ${GLOG_LIBRARIES})


# ------------------------------------------------------------------------------
# Add Torch bindings
# ------------------------------------------------------------------------------
#ADD_SUBDIRECTORY(torch)

ADD_SUBDIRECTORY(include/imgdistort)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)