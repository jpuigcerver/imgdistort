MY_ADD_LIBRARY(cpu_affine
  affine.cc
  affine.h
  affine_generic.h
  COND WITH_IPP affine_ipp.h
  LIBRARIES
  COND WITH_IPP ${IPP_LIBRARIES})

MY_ADD_LIBRARY(cpu_morphology
  morphology.cc
  morphology.h
  morphology_generic.h
  COND WITH_IPP morphology_ipp.h morphology_ipp_operation.h
  LIBRARIES
  COND WITH_IPP ${IPP_LIBRARIES})

IF(WITH_TESTS AND (BUILD_SHARED_LIBS OR BUILD_STATIC_LIBS))
  INCLUDE_DIRECTORIES(${GMOCK_INCLUDE_DIR})
  INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})
  IF(BUILD_SHARED_LIBS)
    SET(affine_lib_ cpu_affine)
    SET(morphology_lib_ cpu_morphology)
  ELSE()
    SET(affine_lib_ cpu_affine_static)
    SET(morphology_lib_ cpu_morphology_static)
  ENDIF()

  ADD_EXECUTABLE(cpu_affine_test
    affine_test.cc
    affine_base_test.cc
    affine_base_test.h
    ../testing/base.h
    ../testing/device_allocator.h
    ../testing/device_allocator_cpu.h
    ../testing/image.h)
  TARGET_LINK_LIBRARIES(cpu_affine_test
    ${affine_lib_}
    ${GLOG_LIBRARIES} ${GMOCK_BOTH_LIBRARIES} ${GTEST_LIBRARIES})
  ADD_TEST(cpu_affine_test cpu_affine_test)
  ADD_IWYU_TO_TARGET(cpu_affine_test)

  ADD_EXECUTABLE(cpu_morphology_test
    morphology_test.cc
    ../testing/base.h
    ../testing/device_allocator.h
    ../testing/device_allocator_cpu.h
    ../testing/image.h
    ../testing/morphology_base.h)
  TARGET_LINK_LIBRARIES(cpu_morphology_test
    ${morphology_lib_}
    ${GLOG_LIBRARIES} ${GMOCK_BOTH_LIBRARIES} ${GTEST_LIBRARIES})
  ADD_TEST(cpu_morphology_test cpu_morphology_test)
  ADD_IWYU_TO_TARGET(cpu_morphology_test)
ENDIF(WITH_TESTS AND (BUILD_SHARED_LIBS OR BUILD_STATIC_LIBS))
