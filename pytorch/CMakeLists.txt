ADD_SUBDIRECTORY(src)

SET(PYTORCH_SETUP_PREFIX ""
  CACHE STRING "Prefix path passed to PyTorch's binding setup.py.")

IF(WITH_PYTORCH)
  SET(_PY_EXTERNAL_LIBS)
  SET(_PY_EXTERNAL_DIRS)
  SET(_PY_HEADERS)

  IF(PYTORCH_CUDA)
    SET(_PY_WITH_CUDA True)
  ELSE()
    SET(_PY_WITH_CUDA False)
  ENDIF(PYTORCH_CUDA)

  # List of headers for the PyTorch library.
  LIST(APPEND _PY_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/affine.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/morphology.h)
  IF(PYTORCH_CUDA)
    LIST(APPEND _PY_HEADERS
      ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/affine.h
      ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/morphology.h)
  ENDIF(PYTORCH_CUDA)

  # Dependency with the wrapper code.
  LIST(APPEND _PY_EXTERNAL_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/src/cpu/libpytorch_cpu_.a)
  IF(PYTORCH_CUDA)
    LIST(APPEND _PY_EXTERNAL_LIBS
      ${CMAKE_CURRENT_BINARY_DIR}/src/gpu/libpytorch_gpu_.a)
  ENDIF(PYTORCH_CUDA)

  # Dependency with other libraries needed by the cpu and gpu implementations.
  GET_PROPERTY(_AUX_LIBS TARGET pytorch_cpu_
    PROPERTY LINK_LIBRARIES)
  GET_PROPERTY(_AUX_DIRS TARGET pytorch_cpu_
    PROPERTY INCLUDE_DIRECTORIES)
  LIST(APPEND _PY_EXTERNAL_LIBS "${_AUX_LIBS}")
  LIST(APPEND _PY_EXTERNAL_DIRS "${_AUX_DIRS}")
  IF(PYTORCH_CUDA)
    GET_PROPERTY(_AUX_LIBS TARGET pytorch_gpu_
      PROPERTY LINK_LIBRARIES)
    GET_PROPERTY(_AUX_DIRS TARGET pytorch_gpu_
      PROPERTY INCLUDE_DIRECTORIES)
    LIST(APPEND _PY_EXTERNAL_LIBS "${_AUX_LIBS}")
    LIST(APPEND _PY_EXTERNAL_DIRS "${_AUX_DIRS}")
  ENDIF(PYTORCH_CUDA)

  # A single target that compiles both pytorch_cpu_ and pytorch_gpu_
  ADD_CUSTOM_TARGET(pytorch_ ALL DEPENDS pytorch_cpu_)
  IF(PYTORCH_CUDA)
    ADD_DEPENDENCIES(pytorch_ pytorch_gpu_)
  ENDIF(PYTORCH_CUDA)

  # Copy files to binary dir
  SET(_COPY_FILES)
  LIST(APPEND _COPY_FILES imgdistort_pytorch/imgdistort.py)
  LIST(APPEND _COPY_FILES imgdistort_pytorch/affine_test.py)
  FOREACH(_file IN LISTS _COPY_FILES)
    CONFIGURE_FILE(
      "${CMAKE_CURRENT_SOURCE_DIR}/${_file}"
      "${CMAKE_CURRENT_BINARY_DIR}/${_file}"
      COPYONLY)
  ENDFOREACH(_file IN LISTS _COPY_FILES)

  # Configure files to binary dir
  SET(INIT_PY_IN
    "${CMAKE_CURRENT_SOURCE_DIR}/imgdistort_pytorch/__init__.py.in")
  SET(INIT_PY "${CMAKE_CURRENT_BINARY_DIR}/imgdistort_pytorch/__init__.py")
  SET(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
  SET(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
  SET(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

  # Set package name.
  IF(PYTORCH_CUDA)
    SET(IMGDISTORT_PYTORCH_PACKAGE
      "imgdistort_pytorch_cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}")
  ELSE()
    SET(IMGDISTORT_PYTORCH_PACKAGE imgdistort_pytorch)
  ENDIF(PYTORCH_CUDA)

  CONFIGURE_FILE("${INIT_PY_IN}" "${INIT_PY}")
  CONFIGURE_FILE("${SETUP_PY_IN}" "${SETUP_PY}")
  ADD_CUSTOM_COMMAND(OUTPUT "${OUTPUT}"
    COMMAND ${PYTHON_EXECUTABLE} setup.py build
    COMMAND ${CMAKE_COMMAND} -E touch "${OUTPUT}"
    DEPENDS "${INIT_PY}"
    DEPENDS "${SETUP_PY}"
    DEPENDS imgdistort_pytorch/imgdistort.py
    DEPENDS imgdistort_pytorch/affine_test.py
    DEPENDS pytorch_
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  # Custom target to run setup.py build during make
  ADD_CUSTOM_TARGET(imgdistort_pytorch ALL DEPENDS "${OUTPUT}")

  # Set PYTHONPATH to run the tests without installing
  SET(_PYTHONPATH
    "${CMAKE_CURRENT_BINARY_DIR}/build/lib.linux-x86_64-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
  SET(_PYTHONPATH "${_PYTHONPATH}:$ENV{PYTHONPATH}")

  # Add Python tests
  ADD_TEST(NAME pytorch_affine_test
    COMMAND ${PYTHON_EXECUTABLE} -m unittest
    imgdistort_pytorch.affine_test)
  SET_TESTS_PROPERTIES(
    pytorch_affine_test
    PROPERTIES
    ENVIRONMENT PYTHONPATH=${_PYTHONPATH}
    DEPENDS imgdistort_pytorch
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # If no PYTORCH_SETUP_PREFIX variable is given by the user, we will use
  # CMAKE_INSTALL_PREFIX (if given).
  IF(NOT PYTORCH_SETUP_PREFIX)
    SET(PYTORCH_SETUP_PREFIX "${CMAKE_INSTALL_PREFIX}")
  ENDIF()

    # make install will just execute python setup.py.
  # The install prefix will be PYTORCH_SETUP_PREFIX, if given. Or the system's
  # default if not given.
  IF(PYTORCH_SETUP_PREFIX)
    INSTALL(
      CODE "EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} setup.py install --prefix=${PYTORCH_SETUP_PREFIX} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
  ELSE()
    INSTALL(
      CODE "EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
  ENDIF(PYTORCH_SETUP_PREFIX)

  SET_PROPERTY(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/build")

ENDIF(WITH_PYTORCH)
