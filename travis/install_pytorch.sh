#!/bin/bash
set -e;

if [ "$TRAVIS_OS_NAME" = linux ]; then
  if [ "$TRAVIS_PYTHON_VERSION" = "2.7" ]; then
    pip install http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl;
  elif [ "$TRAVIS_PYTHON_VERSION" = "3.5" ]; then
    pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl;
  elif [ "$TRAVIS_PYTHON_VERSION" = "3.6" ]; then
    pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl;
  fi;
elif [ "$TRAVIS_OS_NAME" = osx ]; then
  if [ "$TRAVIS_PYTHON_VERSION" = "2.7" ]; then
    pip install http://download.pytorch.org/whl/torch-0.3.0.post4-cp27-none-macosx_10_6_x86_64.whl;
  elif [ "$TRAVIS_PYTHON_VERSION" = "3.5" ]; then
    pip3 install http://download.pytorch.org/whl/torch-0.3.0.post4-cp35-cp35m-macosx_10_6_x86_64.whl;
  elif [ "$TRAVIS_PYTHON_VERSION" = "3.6" ]; then
    pip3 install http://download.pytorch.org/whl/torch-0.3.0.post4-cp36-cp36m-macosx_10_7_x86_64.whl;
  fi;
fi;
