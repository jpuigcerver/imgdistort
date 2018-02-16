#[[
IF(WITH_TESTS AND BUILD_SHARED_LIBS OR BUILD_STATIC_LIBS)

  IF(BUILD_SHARED_LIBS)
    SET(affine_cpu_lib_ imgdistort_cpu_affine)
    SET(affine_gpu_lib_ imgdistort_gpu_affine)
  ELSE()
    SET(affine_cpu_lib_ imgdistort_cpu_affine_static)
    SET(affine_gpu_lib_ imgdistort_gpu_affine_static)
  ENDIF()

  ########################################################################
  ## Affine tests
  ########################################################################
  ADD_LIBRARY(affine_base_test_cpu
    affine_base_test_cpu.cc affine_base_test_cpu.h)
  ADD_IWYU_TO_TARGET(affine_base_test_cpu)

  IF(WITH_CUDA)
    CUDA_ADD_LIBRARY(affine_base_test_gpu
      affine_base_test_gpu.cu affine_base_test_gpu.h)
    TARGET_LINK_LIBRARIES(affine_base_test_gpu affine_base_test_cpu)
    ADD_IWYU_TO_TARGET(affine_base_test_gpu)

    CUDA_ADD_EXECUTABLE(affine_gpu_test affine_gpu_test.cu)
    TARGET_LINK_LIBRARIES(affine_gpu_test
      affine_base_test_gpu ${affine_gpu_lib_}
      ${GLOG_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES})
    ADD_TEST(affine_gpu_test affine_gpu_test)
    ADD_IWYU_TO_TARGET(affine_gpu_test)
  ENDIF(WITH_CUDA)


  ########################################################################
  ## Morphology tests
  ########################################################################
  ADD_EXECUTABLE(morphology_cpu_test morphology_cpu_test.cc)
  TARGET_LINK_LIBRARIES(morphology_cpu_test
    morphology_cpu_static
    ${GLOG_LIBRARIES} ${GMOCK_MAIN_LIBRARIES} ${GTEST_LIBRARIES})
  ADD_TEST(morphology_cpu_test morphology_cpu_test)
  ADD_IWYU_TO_TARGET(morphology_cpu_test)

  IF (WITH_CUDA)
    ADD_EXECUTABLE(morphology_gpu_test morphology_gpu_test.cc)
    TARGET_LINK_LIBRARIES(morphology_gpu_test
      morphology_gpu_static
      ${GLOG_LIBRARIES} ${GMOCK_MAIN_LIBRARIES} ${GTEST_LIBRARIES})
    ADD_TEST(morphology_gpu_test morphology_gpu_test)
    ADD_IWYU_TO_TARGET(morphology_gpu_test)
  ENDIF(WITH_CUDA)
ENDIF()
]]
