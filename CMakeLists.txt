CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(imgdistortion CXX)

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

ENABLE_TESTING()

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Glog REQUIRED)
FIND_PACKAGE(GTest)
FIND_PACKAGE(GMock)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_FORCE_INLINES")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")

FIND_PROGRAM(IWYU_PATH NAMES iwyu include-what-you-use)
IF(IWYU_PATH)
  MESSAGE(STATUS "Found IWYU = ${IWYU_PATH}")
ENDIF()

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")

CUDA_ADD_LIBRARY(affine affine.cu affine.h utils.h STATIC)
TARGET_LINK_LIBRARIES(affine ${GLOG_LIBRARIES})

CUDA_ADD_LIBRARY(morphology morphology.cu morphology.h utils.h STATIC)
TARGET_LINK_LIBRARIES(morphology ${GLOG_LIBRARIES})

IF(IWYU_PATH AND VERSION VERSION_GREATER 3.2)
  SET_PROPERTY(TARGET affine morphology PROPERTY
  CXX_INCLUDE_WHAT_YOU_USE ${IWYU_PATH})
ENDIF()

ADD_SUBDIRECTORY(torch)

IF(GTEST_FOUND AND GMOCK_FOUND)
  ADD_LIBRARY(base_test base_test.cc base_test.h utils.h)
  TARGET_LINK_LIBRARIES(base_test ${GLOG_LIBRARIES})

  ADD_EXECUTABLE(affine_test affine_test.cc utils.h)
  TARGET_LINK_LIBRARIES(affine_test
    affine base_test ${GLOG_LIBRARIES} ${GMOCK_MAIN_LIBRARIES})
  GTEST_ADD_TESTS(affine_test "" affine_test.cc)

  ADD_EXECUTABLE(morphology_test morphology_test.cc utils.h)
  TARGET_LINK_LIBRARIES(morphology_test
    morphology base_test ${GLOG_LIBRARIES} ${GMOCK_MAIN_LIBRARIES})
  GTEST_ADD_TESTS(morphology_test "" morphology_test.cc)


  IF(IWYU_PATH AND VERSION VERSION_GREATER 3.2)
    SET_PROPERTY(TARGET affine_test morphology_test PROPERTY
    CXX_INCLUDE_WHAT_YOU_USE ${IWYU_PATH})
  ENDIF()
ELSE()
  MESSAGE(STATUS "Missing GMock or GTest, tests will not be compiled!")
ENDIF()
