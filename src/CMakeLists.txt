MACRO(ADD_SHARED_AND_STATIC_LIBRARY name)
  SET(srcs_)
  SET(libs_)
  SET(found_link_libraries_ FALSE)

  FOREACH(arg_ ${ARGN})
    IF("x${arg_}" STREQUAL "xLINK_LIBRARIES")
      SET(found_link_libraries_ TRUE)
    ELSE()
      IF(found_link_libraries_)
	LIST(APPEND libs_ ${arg_})
      ELSE()
	LIST(APPEND srcs_ ${arg_})
      ENDIF()
    ENDIF()
  ENDFOREACH()

  ADD_LIBRARY(${name}_obj OBJECT "${srcs_}")
  ADD_LIBRARY(${name} SHARED $<TARGET_OBJECTS:${name}_obj>)
  ADD_LIBRARY(${name}_static STATIC $<TARGET_OBJECTS:${name}_obj>)

  TARGET_LINK_LIBRARIES(${name} "${libs_}")
  TARGET_LINK_LIBRARIES(${name}_static "${libs_}")
ENDMACRO(ADD_SHARED_AND_STATIC_LIBRARY)

MACRO(CUDA_ADD_SHARED_AND_STATIC_LIBRARY name)
  SET(srcs_)
  SET(libs_)
  SET(found_link_libraries_ FALSE)

  FOREACH(arg_ ${ARGN})
    IF("x${arg_}" STREQUAL "xLINK_LIBRARIES")
      SET(found_link_libraries_ TRUE)
    ELSE()
      IF(found_link_libraries_)
	LIST(APPEND libs_ ${arg_})
      ELSE()
	LIST(APPEND srcs_ ${arg_})
      ENDIF()
    ENDIF()
  ENDFOREACH()

  CUDA_COMPILE(objs_ "${srcs_}")
  CUDA_ADD_LIBRARY(${name} SHARED "${objs_}")
  CUDA_ADD_LIBRARY(${name}_static STATIC "${objs_}")

  TARGET_LINK_LIBRARIES(${name} "${libs_}")
  TARGET_LINK_LIBRARIES(${name}_static "${libs_}")
ENDMACRO(CUDA_ADD_SHARED_AND_STATIC_LIBRARY)

IF(IPP_FOUND AND WITH_IPP)
  ADD_SHARED_AND_STATIC_LIBRARY(affine_cpu
    affine_cpu.cc
    ../include/imgdistort/affine_cpu.h
    ../include/imgdistort/affine_cpu_generic.h
    ../include/imgdistort/affine_cpu_ipp.h
    LINK_LIBRARIES ${IPP_LIBRARIES} ${GLOG_LIBRARIES})
  ADD_SHARED_AND_STATIC_LIBRARY(morphology_cpu
    morphology_cpu.cc
    ../include/imgdistort/morphology_cpu.h
    ../include/imgdistort/morphology_cpu_generic.h
    ../include/imgdistort/morphology_cpu_ipp.h
    LINK_LIBRARIES ${IPP_LIBRARIES} ${GLOG_LIBRARIES})
ELSE()
  ADD_SHARED_AND_STATIC_LIBRARY(affine_cpu
    affine_cpu.cc
    ../include/imgdistort/affine_cpu.h
    ../include/imgdistort/affine_cpu_generic.h
    LINK_LIBRARIES ${GLOG_LIBRARIES})
  ADD_SHARED_AND_STATIC_LIBRARY(morphology_cpu
    morphology_cpu.cc
    ../include/imgdistort/morphology_cpu.h
    ../include/imgdistort/morphology_cpu_generic.h
    LINK_LIBRARIES ${GLOG_LIBRARIES})
ENDIF()

IF(CUDA_FOUND AND WITH_CUDA)
  CUDA_ADD_SHARED_AND_STATIC_LIBRARY(affine_gpu
    affine_gpu.cu
    LINK_LIBRARIES ${GLOG_LIBRARIES})
  CUDA_ADD_SHARED_AND_STATIC_LIBRARY(morphology_gpu
    morphology_gpu.cu
    LINK_LIBRARIES ${GLOG_LIBRARIES})
ENDIF()
