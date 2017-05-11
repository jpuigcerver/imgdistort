INSTALL(FILES affine_cpu.h DESTINATION include/imgdistort)
INSTALL(FILES morphology_cpu.h DESTINATION include/imgdistort)

IF(CUDA_FOUND AND WITH_CUDA)
  INSTALL(FILES affine_gpu.h DESTINATION include/imgdistort)
  INSTALL(FILES morphology_gpu.h DESTINATION include/imgdistort)
ENDIF()