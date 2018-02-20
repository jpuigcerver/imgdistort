#!/bin/bash
set -e;

mkdir build;
cd build;
cmake -DWITH_CUDA=OFF -DWITH_GLOG=OFF -DWITH_IPP=OFF -DWITH_PYTORCH=ON \
      -DWITH_TESTS=ON -DCMAKE_BUILD_TYPE=DEBUG ..;
make VERBOSE=1;

cd pytorch;
python setup.py bdist_wheel;
pip install $(find dist/ -name "*.whl");
