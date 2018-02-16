IF(WITH_CUDA)

MY_CUDA_ADD_LIBRARY(gpu_affine
  affine.cu
  affine.h)

MY_CUDA_ADD_LIBRARY(gpu_morphology
  morphology.cu
  morphology.h)

  IF(WITH_TESTS AND (BUILD_SHARED_LIBS OR BUILD_STATIC_LIBS))
    IF(BUILD_SHARED_LIBS)
      SET(affine_lib_ gpu_affine)
      SET(morphology_lib_ gpu_morphology)
    ELSE()
      SET(affine_lib_ gpu_affine_static)
      SET(morphology_lib_ gpu_morphology_static)
    ENDIF()

    CUDA_ADD_EXECUTABLE(gpu_affine_test
      affine_test.cu
      affine_base_test.cu
      affine_base_test.h
      ../cpu/affine_base_test.cc
      ../cpu/affine_base_test.h
      ../testing/base.h
      ../testing/device_allocator.h
      ../testing/device_allocator_gpu.h
      ../testing/image.h)
    TARGET_LINK_LIBRARIES(gpu_affine_test
      ${affine_lib_}
      ${GLOG_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES})
    ADD_TEST(gpu_affine_test gpu_affine_test)
    ADD_IWYU_TO_TARGET(gpu_affine_test)

    CUDA_ADD_EXECUTABLE(gpu_morphology_test
      morphology_test.cu
      ../testing/base.h
      ../testing/device_allocator.h
      ../testing/device_allocator_gpu.h
      ../testing/image.h
      ../testing/morphology_base.h)
    TARGET_LINK_LIBRARIES(gpu_morphology_test
      ${morphology_lib_}
      ${GLOG_LIBRARIES} ${GMOCK_MAIN_LIBRARIES} ${GTEST_LIBRARIES})
    ADD_TEST(gpu_morphology_test gpu_morphology_test)
    ADD_IWYU_TO_TARGET(gpu_morphology_test)
  ENDIF(WITH_TESTS AND (BUILD_SHARED_LIBS OR BUILD_STATIC_LIBS))

ENDIF(WITH_CUDA)
