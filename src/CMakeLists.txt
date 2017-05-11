# Affine transformations
IF(IPP_FOUND AND WITH_IPP)
  ADD_LIBRARY(affine_cpu affine_cpu.cc affine_cpu_generic.cc affine_cpu_ipp.cc)
  TARGET_LINK_LIBRARIES(affine_cpu ${IPP_LIBRARIES})
ELSE()
  ADD_LIBRARY(affine_cpu affine_cpu.cc affine_cpu_generic.cc)
ENDIF()

IF(CUDA_FOUND AND WITH_CUDA)
  CUDA_ADD_LIBRARY(affine_gpu affine_gpu.cu)
ENDIF()


# Morphology transformations
IF (IPP_FOUND AND WITH_IPP)
  ADD_LIBRARY(morphology_cpu morphology_cpu.cc morphology_cpu_generic.cc
    morphology_cpu_ipp.cc)
ELSE()
  ADD_LIBRARY(morphology_cpu morphology_cpu.cc morphology_cpu_generic.cc)
ENDIF()

IF(CUDA_FOUND AND WITH_CUDA)
  CUDA_ADD_LIBRARY(morphology_gpu morphology_gpu.cu)
ENDIF()