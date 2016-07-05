FIND_PACKAGE(Torch)

IF (Torch_FOUND)
  MESSAGE(STATUS "Building Torch bindings")
  INCLUDE_DIRECTORIES(
    ${Torch_INSTALL_INCLUDE}
    ${Torch_INSTALL_INCLUDE}/TH
    ${Torch_INSTALL_INCLUDE}/THC
    "${CMAKE_CURRENT_SOURCE_DIR}")

  ADD_TORCH_PACKAGE(torch_imgdistort affine_binding.cu utils.c init.lua)
  TARGET_LINK_LIBRARIES(torch_imgdistort luajit luaT THC TH)
ENDIF()