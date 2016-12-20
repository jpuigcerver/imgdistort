FIND_PACKAGE(GTest)
FIND_PACKAGE(GMock)
FIND_PACKAGE(ImageMagick COMPONENTS Magick++ REQUIRED)

IF(GTEST_FOUND AND GMOCK_FOUND)
  INCLUDE_DIRECTORIES(${ImageMagick_Magick++_INCLUDE_DIRS})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16")
  # ADD_LIBRARY(base_test base_test.cc base_test.h utils.h)
  # TARGET_LINK_LIBRARIES(base_test ${GLOG_LIBRARIES})

  ADD_EXECUTABLE(affine_test affine_test.cc)
  TARGET_LINK_LIBRARIES(affine_test
    affine_cpu ${IPP_LIBRARIES} ${GLOG_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES})
  TARGET_LINK_LIBRARIES(affine_test ${ImageMagick_Magick++_LIBRARY})
  #   affine base_test ${GLOG_LIBRARIES} ${GMOCK_MAIN_LIBRARIES})
  # GTEST_ADD_TESTS(affine_test "" affine_test.cc)

  # ADD_EXECUTABLE(morphology_test morphology_test.cc utils.h)
  # TARGET_LINK_LIBRARIES(morphology_test
  #   morphology base_test ${GLOG_LIBRARIES} ${GMOCK_MAIN_LIBRARIES})
  # GTEST_ADD_TESTS(morphology_test "" morphology_test.cc)


  # IF(IWYU_PATH AND VERSION VERSION_GREATER 3.2)
  #   SET_PROPERTY(TARGET affine_test morphology_test PROPERTY
  #   CXX_INCLUDE_WHAT_YOU_USE ${IWYU_PATH})
  # ENDIF()
ELSE()
  MESSAGE(STATUS "Missing GMock or GTest, tests will not be compiled!")
ENDIF()
