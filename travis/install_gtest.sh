#!/bin/bash

if [ -n "$GTEST_ROOT" ]; then
  ## Install Google Test & Google Mock from sources
  git clone https://github.com/google/googletest
  cd googletest
  mkdir -p build && cd build
  cmake -DBUILD_GTEST=OFF -DBUILD_GMOCK=ON -DCMAKE_INSTALL_PREFIX=$GTEST_ROOT ..
  make -j4
  make install
fi;
