FIND_PACKAGE(Torch)

IF (Torch_FOUND)
  MESSAGE(STATUS "Building Torch bindings")
  INCLUDE_DIRECTORIES(
    ${Torch_INSTALL_INCLUDE}
    ${Torch_INSTALL_INCLUDE}/TH
    ${Torch_INSTALL_INCLUDE}/THC
    "${CMAKE_CURRENT_SOURCE_DIR}")

  SET(src affine_binding.cu utils.c)
  SET(luasrc init.lua)
  ADD_TORCH_PACKAGE(torch_imgdistort "${src}" "${luasrc}")
  TARGET_LINK_LIBRARIES(torch_imgdistort luajit luaT THC TH)

  INSTALL(TARGETS torch_imgdistort
    RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
    LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
    ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

ENDIF()