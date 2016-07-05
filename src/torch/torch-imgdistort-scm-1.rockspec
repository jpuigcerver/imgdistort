package = "torch-imgdistort"
version = "scm-1"

source = {
   url = "git://github.com/jpuigcerver/imgdistort.git",
}

description = {
   summary = "",
   detailed = [[
   ]],
   homepage = "https://github.com/jpuigcerver/imgdistort",
   license = "MIT"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DLUALIB=$(LUALIB) -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN) && make install
]],
  platforms = {},
  install_command = "cd build"
}