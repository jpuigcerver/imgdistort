# TODO(jpuigcerver): These tests will only work if PyTorch is
# installed. Allow to use the TH and THC in Torch as well.

IF(WITH_TESTS AND (WITH_PYTORCH OR WITH_TORCH))
  INCLUDE_DIRECTORIES(${GMOCK_INCLUDE_DIR})
  INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})

  IF(WITH_PYTORCH)
    SET(_th_incl "${PYTORCH_INCLUDE_DIR}")
    LIST(APPEND _th_incl "${PYTORCH_INCLUDE_DIR}/TH")
    SET(_th_lib "${PYTORCH_TH_LIBRARY}")
    IF(WITH_CUDA AND PYTORCH_THC_LIBRARY)
      SET(_thc_incl "${PYTORCH_INCLUDE_DIR}/THC")
      SET(_thc_lib "${PYTORCH_THC_LIBRARY}")
    ENDIF(WITH_CUDA AND PYTORCH_THC_LIBRARY)
  ELSE()
    SET(_th_incl "${TORCH_INCLUDE_DIR}")
    LIST(APPEND _th_incl "${TORCH_INCLUDE_DIR}/TH")
    SET(_th_lib "${TORCH_TH_LIBRARY}")
    IF(WITH_CUDA AND TORCH_THC_LIBRARY)
      SET(_thc_incl "${TORCH_INCLUDE_DIR}/THC")
      SET(_thc_lib "${TORCH_THC_LIBRARY}")
    ENDIF(WITH_CUDA AND TORCH_THC_LIBRARY)
  ENDIF()

  INCLUDE_DIRECTORIES("${_th_incl}")
  ADD_EXECUTABLE(THTensorTest
    THTensorTest.cc
    THTensor.h THTraits.h generic/THTensor.h)
  TARGET_LINK_LIBRARIES(THTensorTest
    "${_th_lib}"
    "${GTEST_BOTH_LIBRARIES}")
  ADD_TEST(THTensorTest THTensorTest)

  IF(WITH_CUDA AND (PYTORCH_THC_LIBRARY OR TORCH_THC_LIBRARY))
    INCLUDE_DIRECTORIES("${_thc_incl}")
    CUDA_ADD_EXECUTABLE(THCTensorTest
      THCTensorTest.cc
      THTensor.h THTraits.h generic/THTensor.h
      THCTensor.h THCTraits.h generic/THCTensor.h)
    TARGET_LINK_LIBRARIES(THCTensorTest
      "${_th_lib}"
      "${_thc_lib}"
      "${GTEST_BOTH_LIBRARIES}")
    ADD_TEST(THCTensorTest THCTensorTest)
  ENDIF(WITH_CUDA AND (PYTORCH_THC_LIBRARY OR TORCH_THC_LIBRARY))

ENDIF(WITH_TESTS AND (WITH_PYTORCH OR WITH_TORCH))
