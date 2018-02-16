IF(WITH_PYTORCH)
  INCLUDE_DIRECTORIES(
    "${PYTORCH_INCLUDE_DIR}"
    "${PYTORCH_INCLUDE_DIR}/TH")
  ADD_LIBRARY(pytorch_cpu_ STATIC
    ${PROJECT_SOURCE_DIR}/torch/affine.h
    affine.cc
    affine.h

    ${PROJECT_SOURCE_DIR}/torch/morphology.h
    morphology.cc
    morphology.h)
  TARGET_LINK_LIBRARIES(pytorch_cpu_
    ${PYTORCH_TH_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
    m)
ENDIF(WITH_PYTORCH)
