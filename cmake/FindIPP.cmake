#
# The script to detect Intel(R) Integrated Performance Primitives (IPP)
# installation/package. Based on OpenCV's FindIPP.
#
# IPP_ROOT_DIR      - Folder containing the IPP root dir (e.g. /opt/intel/ipp)
#
# IPP_FOUND         - True if IPP was found
# IPP_INCLUDE_DIRS  - IPP include directory
# IPP_LIBRARIES     - IPP libraries needed for linking
# IPP_VERSION_STR   - String with the newest detected IPP version
# IPP_VERSION_MAJOR - Numbers of IPP version (MAJOR.MINOR.BUILD)
# IPP_VERSION_MINOR
# IPP_VERSION_BUILD

INCLUDE(FindPackageHandleStandardArgs)

SET(IPP_ROOT_DIR ""
  CACHE PATH "Folder containing the root Torch install directory")

FIND_PATH(IPP_INCLUDE_DIR
  NAMES ipp.h
  HINTS ${IPP_ROOT_DIR}/include)
MARK_AS_ADVANCED(IPP_INCLUDE_DIR)

# This function detects IPP version by analyzing .h file
macro(ipp_get_version VERSION_FILE)
  unset(_VERSION_STR)
  unset(_MAJOR)
  unset(_MINOR)
  unset(_BUILD)

  # read IPP version info from file
  file(STRINGS ${VERSION_FILE} STR1 REGEX "IPP_VERSION_MAJOR")
  file(STRINGS ${VERSION_FILE} STR2 REGEX "IPP_VERSION_MINOR")
  file(STRINGS ${VERSION_FILE} STR3 REGEX "IPP_VERSION_BUILD")
  if("${STR3}" STREQUAL "")
    file(STRINGS ${VERSION_FILE} STR3 REGEX "IPP_VERSION_UPDATE")
  endif()
  file(STRINGS ${VERSION_FILE} STR4 REGEX "IPP_VERSION_STR")

  # extract info and assign to variables
  string(REGEX MATCHALL "[0-9]+" _MAJOR ${STR1})
  string(REGEX MATCHALL "[0-9]+" _MINOR ${STR2})
  string(REGEX MATCHALL "[0-9]+" _BUILD ${STR3})
  string(REGEX MATCHALL "[0-9]+[.]+[0-9]+[^\"]+|[0-9]+[.]+[0-9]+" _VERSION_STR ${STR4})

  # export info to parent scope
  set(IPP_VERSION_STR   ${_VERSION_STR})
  set(IPP_VERSION_MAJOR ${_MAJOR})
  set(IPP_VERSION_MINOR ${_MINOR})
  set(IPP_VERSION_BUILD ${_BUILD})
endmacro()

IF(IPP_INCLUDE_DIR)
  ipp_get_version(${IPP_INCLUDE_DIR}/ippversion.h)
ENDIF()

# Get architecture
SET(IPP_X64 0)
IF(CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 8)
  SET(IPP_X64 1)
ENDIF()
IF(CMAKE_CL_64)
  SET(IPP_X64 1)
ENDIF()

# Determine hint directory to find the libraries
SET(_IPP_LIB_DIR_HINT)
IF(IPP_ROOT_DIR)
  IF(APPLE)
    SET(_IPP_LIB_DIR_HINT ${IPP_ROOT_DIR}/lib)
  ELSEIF(IPP_X64)
    SET(_IPP_LIB_DIR_HINT ${IPP_ROOT_DIR}/lib/intel64)
  ELSE()
    SET(_IPP_LIB_DIR_HINT ${IPP_ROOT_DIR}/lib/ia32)
  ENDIF()
ENDIF()

FIND_LIBRARY(IPP_LIB_CORE NAMES ippcore HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_AC   NAMES ippac   HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_CC   NAMES ippcc   HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_CH   NAMES ippch   HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_CV   NAMES ippcv   HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_DC   NAMES ippdc   HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_DI   NAMES ippdi   HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_GEN  NAMES ippgen  HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_I    NAMES ippi    HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_J    NAMES ippj    HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_M    NAMES ippm    HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_R    NAMES ippr    HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_S    NAMES ipps    HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_SC   NAMES ippsc   HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_VC   NAMES ippvc   HINTS ${_IPP_LIB_DIR_HINT})
FIND_LIBRARY(IPP_LIB_VM   NAMES ippvm   HINTS ${_IPP_LIB_DIR_HINT})
MARK_AS_ADVANCED(IPP_LIB_CORE IPP_LIB_AC IPP_LIB_CC IPP_LIB_CH IPP_LIB_CV
  IPP_LIB_DC IPP_LIB_DI IPP_LIB_GEN IPP_LIB_I IPP_LIB_J IPP_LIB_M
  IPP_LIB_R IPP_LIB_S IPP_LIB_SC IPP_LIB_VC IPP_LIB_VM)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(IPP
  DEFAULT_MSG IPP_INCLUDE_DIR IPP_LIB_CORE IPP_LIB_I IPP_LIB_S)

IF (IPP_FOUND)
  MESSAGE(STATUS "Found IPP version ${IPP_VERSION_STR}")
  SET(IPP_INCLUDE_DIRS ${IPP_INCLUDE_DIR})
  SET(IPP_LIBRARIES ${IPP_LIB_CORE} ${IPP_LIB_I} ${IPP_LIB_S} ${IPP_LIB_CV})
ENDIF()
