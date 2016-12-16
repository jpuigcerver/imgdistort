package = "imgdistort"
version = "scm-1"

source = {
   url = "git://github.com/jpuigcerver/imgdistort.git",
}

description = {
   summary = "Torch7 bindings for CUDA image distortions",
   detailed = [[
   ]],
   homepage = "https://github.com/jpuigcerver/imgdistort",
   license = "MIT"
}

dependencies = {
   "torch >= 7.0",
   "cutorch",
}

build = {
  type = "cmake",
  variables = {
    TORCH_ROOT = "$(LUA_BINDIR)/..",
    WITH_LUAROCKS = "ON",
    CMAKE_BUILD_TYPE = "RELEASE",
    CMAKE_INSTALL_PREFIX = "$(PREFIX)",
    INST_LIBDIR="$(LIBDIR)",
    INST_LUADIR="$(LUADIR)",
  }
}