#!/bin/bash
set -e;

# Test module in the build directory.
cd build;
make test;

# Test module installed via pip.
python -m unittest imgdistort_pytorch.affine_test;
python -m unittest imgdistort_pytorch.morphology_test;
