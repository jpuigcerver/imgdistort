find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

CUDA_ADD_EXECUTABLE(affine_test
  affine_test.cu
  affine.cuh base_test.cuh utils.cuh)
TARGET_LINK_LIBRARIES(affine_test gflags glog gtest gmock)

ADD_SUBDIRECTORY(torch)
