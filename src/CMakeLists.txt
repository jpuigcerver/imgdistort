find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -Xcompiler -fPIC")

CUDA_ADD_LIBRARY(affine affine.cu affine.h utils.h STATIC)
TARGET_LINK_LIBRARIES(affine glog)

CUDA_ADD_LIBRARY(morphology morphology.cu morphology.h utils.h STATIC)
TARGET_LINK_LIBRARIES(morphology glog)

# Tests
ADD_LIBRARY(base_test base_test.cc base_test.h utils.h)
TARGET_LINK_LIBRARIES(base_test glog gtest gmock)
TARGET_COMPILE_OPTIONS(base_test PUBLIC "-std=c++11")

ADD_EXECUTABLE(affine_test affine_test.cc utils.h)
TARGET_LINK_LIBRARIES(affine_test
  affine base_test
  gflags glog gtest gmock)
TARGET_COMPILE_OPTIONS(affine_test PUBLIC "-std=c++11")

ADD_EXECUTABLE(morphology_test morphology_test.cc utils.h)
TARGET_LINK_LIBRARIES(morphology_test
  morphology base_test
  gflags glog gtest gmock)
TARGET_COMPILE_OPTIONS(morphology_test PUBLIC "-std=c++11")


ADD_SUBDIRECTORY(torch)
