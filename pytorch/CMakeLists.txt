IF(PYTORCH_FOUND AND WITH_PYTORCH)
  SET(_EXTERNAL_LIBS)
  SET(_EXTERNAL_DIRS)

  IF(CUDA_FOUND AND WITH_CUDA)
    INCLUDE_DIRECTORIES("${PYTORCH_INCLUDE_DIR}/TH")
    INCLUDE_DIRECTORIES("${PYTORCH_INCLUDE_DIR}/THC")
    CUDA_ADD_LIBRARY(imgdistort_ STATIC
      ../include/imgdistort/affine_cpu.h
      ../include/imgdistort/affine_gpu.h
      ../include/imgdistort/morphology_cpu.h
      ../include/imgdistort/morphology_gpu.h
      src/binding_common.h
      src/binding_cpu.h
      src/binding_gpu.h
      src/binding_gpu.cu
      src/binding_cpu.cc)
  ELSE()
    INCLUDE_DIRECTORIES("${PYTORCH_INCLUDE_DIR}/TH")
    ADD_LIBRARY(imgdistort_ STATIC
      ../include/imgdistort/affine_cpu.h
      ../include/imgdistort/morphology_cpu.h
      src/binding_common.h
      src/binding_cpu.h
      src/binding_cpu.cc)
  ENDIF(CUDA_FOUND AND WITH_CUDA)

  # Dependency with the wrapper code.
  LIST(APPEND _EXTERNAL_LIBS ${CMAKE_CURRENT_BINARY_DIR}/libimgdistort_.a)

  # Dependency with the core imgdistort library.
  LIST(APPEND _EXTERNAL_LIBS
    ${PROJECT_BINARY_DIR}/src/libaffine_cpu_static.a)
  LIST(APPEND _EXTERNAL_LIBS
    ${PROJECT_BINARY_DIR}/src/libmorphology_cpu_static.a)
  IF(CUDA_FOUND AND WITH_CUDA)
    LIST(APPEND _EXTERNAL_LIBS
      ${PROJECT_BINARY_DIR}/src/libaffine_gpu_static.a)
    LIST(APPEND _EXTERNAL_LIBS
      ${PROJECT_BINARY_DIR}/src/libmorphology_gpu_static.a)
  ENDIF(CUDA_FOUND AND WITH_CUDA)

  # Dependency with Intel IPP
  IF(IPP_FOUND AND WITH_IPP)
    LIST(APPEND _EXTERNAL_LIBS ${IPP_LIBRARIES})
  ENDIF(IPP_FOUND AND WITH_IPP)

  # Dependency with CUDA
  IF(CUDA_FOUND AND WITH_CUDA)
    LIST(APPEND _EXTERNAL_LIBS ${PYTORCH_TH_LIBRARY})
    LIST(APPEND _EXTERNAL_LIBS ${PYTORCH_THC_LIBRARY})
    LIST(APPEND _EXTERNAL_LIBS ${CUDA_LIBRARIES})
  ENDIF(CUDA_FOUND AND WITH_CUDA)

  # Dependency with Google Log
  LIST(APPEND _EXTERNAL_LIBS ${GLOG_LIBRARIES})

  # Dependency with threading library
  LIST(APPEND _EXTERNAL_LIBS ${CMAKE_THREAD_LIBS_INIT})

  # Add include directories needed for the Python extension compilation.
  GET_PROPERTY(_EXTERNAL_DIRS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY INCLUDE_DIRECTORIES)

  SET(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
  SET(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
  SET(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

  CONFIGURE_FILE("${SETUP_PY_IN}"  "${SETUP_PY}")
  ADD_CUSTOM_COMMAND(OUTPUT "${OUTPUT}"
    COMMAND ${PYTHON_EXECUTABLE} "${SETUP_PY}" build
    COMMAND ${CMAKE_COMMAND} -E touch "${OUTPUT}"
    DEPENDS imgdistort_pytorch/__init__.py)
  ADD_CUSTOM_TARGET(imgdistort_pytorch ALL DEPENDS "${OUTPUT}" imgdistort_)

  IF(NOT PYTHON_SETUP_PREFIX)
    SET(PYTHON_SETUP_PREFIX "${CMAKE_INSTALL_PREFIX}")
  ENDIF()
  MESSAGE(STATUS ${PYTHON_SETUP_PREFIX})

  IF(PYTHON_SETUP_PREFIX)
    INSTALL(
      CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install --prefix=${PYTHON_SETUP_PREFIX})")
  ELSE()
    INSTALL(
      CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")
  ENDIF(PYTHON_SETUP_PREFIX)



ENDIF(PYTORCH_FOUND AND WITH_PYTORCH)
