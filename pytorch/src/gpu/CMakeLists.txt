IF(PYTORCH_CUDA)
  INCLUDE_DIRECTORIES(
    "${PYTORCH_INCLUDE_DIR}"
    "${PYTORCH_INCLUDE_DIR}/TH"
    "${PYTORCH_INCLUDE_DIR}/THC")
  CUDA_ADD_LIBRARY(pytorch_gpu_ STATIC
    ${PROJECT_SOURCE_DIR}/torch/affine.h
    affine.h
    affine.cu

    ${PROJECT_SOURCE_DIR}/torch/morphology.h
    morphology.h
    morphology.cu)
  TARGET_LINK_LIBRARIES(pytorch_gpu_
    ${PYTORCH_TH_LIBRARY}
    ${PYTORCH_THC_LIBRARY}
    ${CUDA_LIBRARIES})
ENDIF(PYTORCH_CUDA)
