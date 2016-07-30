FIND_PACKAGE(Torch)

IF (Torch_FOUND)
  MESSAGE(STATUS "Building Torch bindings")
  INCLUDE_DIRECTORIES(
    ${Torch_INSTALL_INCLUDE}
    ${Torch_INSTALL_INCLUDE}/TH
    ${Torch_INSTALL_INCLUDE}/THC
    "${CMAKE_CURRENT_SOURCE_DIR}")

  ADD_LIBRARY(imgdistort MODULE binding.cc utils.c utils.h)
  TARGET_LINK_LIBRARIES(imgdistort
    affine morphology
    luajit luaT THC TH gflags glog)

  INSTALL(TARGETS imgdistort
    RUNTIME DESTINATION "${Torch_INSTALL_BIN}"
    LIBRARY DESTINATION "${Torch_INSTALL_LIB}/lua/5.1"
    ARCHIVE DESTINATION "${Torch_INSTALL_LIB}/lua/5.1")
  INSTALL(FILES init.lua
    DESTINATION "${Torch_INSTALL_SHARE}/lua/5.1/imgdistort")
ENDIF()
