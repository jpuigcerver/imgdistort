FIND_PACKAGE(Torch)

IF (Torch_FOUND)
  MESSAGE(STATUS "Building Torch bindings")
  INCLUDE_DIRECTORIES(
    ${Torch_INSTALL_INCLUDE}
    ${Torch_INSTALL_INCLUDE}/TH
    ${Torch_INSTALL_INCLUDE}/THC
    "${CMAKE_CURRENT_SOURCE_DIR}")

  IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing torch-imgdistort through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
  ENDIF()

  SET(src affine_binding.cu utils.c)
  SET(luasrc init.lua)
  ADD_TORCH_PACKAGE(torch_imgdistort "${src}" "${luasrc}")
  TARGET_LINK_LIBRARIES(torch_imgdistort luajit luaT THC TH)
ENDIF()