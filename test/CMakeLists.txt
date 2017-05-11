FIND_PACKAGE(GTest)
FIND_PACKAGE(GMock)

IF(WITH_TESTS)
  IF(GTEST_FOUND AND GMOCK_FOUND)
    ########################################################################
    ## Affine tests
    ########################################################################
    ADD_LIBRARY(affine_base_test_cpu
      affine_base_test_cpu.cc affine_base_test_cpu.h)
    ADD_IWYU_TO_TARGET(affine_base_test_cpu)

    ADD_EXECUTABLE(affine_cpu_test affine_cpu_test.cc)
    TARGET_LINK_LIBRARIES(affine_cpu_test
      affine_base_test_cpu
      affine_cpu
      ${GLOG_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES})
    ADD_TEST(affine_cpu_test affine_cpu_test)
    ADD_IWYU_TO_TARGET(affine_cpu_test)

    IF (CUDA_FOUND AND WITH_CUDA)
      CUDA_ADD_LIBRARY(affine_base_test_gpu
	affine_base_test_gpu.cu affine_base_test_gpu.h)
      TARGET_LINK_LIBRARIES(affine_base_test_gpu affine_base_test_cpu)
      ADD_IWYU_TO_TARGET(affine_base_test_gpu)

      CUDA_ADD_EXECUTABLE(affine_gpu_test affine_gpu_test.cu)
      TARGET_LINK_LIBRARIES(affine_gpu_test
	affine_gpu affine_base_test_gpu
	${GLOG_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES})
      ADD_TEST(affine_gpu_test affine_gpu_test)
      ADD_IWYU_TO_TARGET(affine_gpu_test)
    ENDIF()


    ########################################################################
    ## Morphology tests
    ########################################################################
    ADD_EXECUTABLE(morphology_cpu_test morphology_cpu_test.cc)
    TARGET_LINK_LIBRARIES(morphology_cpu_test
      morphology_cpu
      ${GLOG_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES})
    ADD_TEST(morphology_cpu_test morphology_cpu_test)
    ADD_IWYU_TO_TARGET(morphology_cpu_test)

    IF (CUDA_FOUND AND WITH_CUDA)
      ADD_EXECUTABLE(morphology_gpu_test morphology_gpu_test.cc)
      TARGET_LINK_LIBRARIES(morphology_gpu_test
	morphology_gpu
	${GLOG_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES})
      ADD_TEST(morphology_gpu_test morphology_gpu_test)
      ADD_IWYU_TO_TARGET(morphology_gpu_test)
    ENDIF()
  ELSE()
    MESSAGE(STATUS "Missing GMock or GTest, tests will not be compiled!")
  ENDIF()
ENDIF()
