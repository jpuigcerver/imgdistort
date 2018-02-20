#!/bin/bash
set -e;

if [ "$TRAVIS_OS_NAME" = linux ]; then
  :
elif [ "$TRAVIS_OS_NAME" = osx ]; then
  :
fi;

if [ "$TRAVIS_PYTHON_VERSION" = "2.7" ]; then
  pip install cffi;
  pip install opencv-python;
elif [ "$TRAVIS_PYTHON_VERSION" = "3.5" ]; then
  pip3 install cffi;
  pip3 install opencv-python;
elif [ "$TRAVIS_PYTHON_VERSION" = "3.6" ]; then
  pip3 install cffi;
  pip3 install opencv-python;
fi;
