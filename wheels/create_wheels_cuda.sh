#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
SOURCE_DIR=$(cd $SDIR/.. && pwd);

if [ "$DOCKER" != 1 ]; then
  cd $SDIR;
  CUDA_VERSIONS=(80 90 91);
  CUDA_IMAGES=(
    nvidia/cuda:8.0-devel
	nvidia/cuda:9.0-devel
	nvidia/cuda:9.1-devel);
  for i in $(seq 1 ${#CUDA_VERSIONS[@]}); do
    rm -rf /tmp/imgdistort/wheels/cu${CUDA_VERSIONS[i - 1]};
    mkdir -p /tmp/imgdistort/wheels/cu${CUDA_VERSIONS[i - 1]};
    nvidia-docker build --build-arg BASE_IMAGE=${CUDA_IMAGES[i - 1]} \
	   -t imgdistort:cu${CUDA_VERSIONS[i - 1]}-base -f Dockerfile .;
    nvidia-docker build --build-arg CUDA_VERSION_SHORT=${CUDA_VERSIONS[i - 1]} \
	   -t imgdistort:cu${CUDA_VERSIONS[i - 1]} -f Dockerfile-cuda .;
    docker run --runtime=nvidia --rm --log-driver none \
	   -v /tmp:/host/tmp \
	   -v ${SOURCE_DIR}:/host/src \
	   imgdistort:cu${CUDA_VERSIONS[i - 1]} /create_wheels_cuda.sh;
  done;
  exit 0;
fi;

## THIS CODE IS EXECUTED WITHIN THE DOCKER CONTAINER

# Copy source in the host to a temporal location.
cp -r /host/src /tmp/src;
cd /tmp/src;
git status;

if [ "$CUDA_VERSION_SHORT" = 75 ]; then
  export CUDA_ARCH_LIST="Kepler Maxwell";
elif [ "$CUDA_VERSION_SHORT" = 80 ]; then
  export CUDA_ARCH_LIST="Kepler Maxwell Pascal";
elif  [ "$CUDA_VERSION_SHORT" = 90 ]; then
  export CUDA_ARCH_LIST="Kepler Maxwell Pascal 7.0+PTX";
elif  [ "$CUDA_VERSION_SHORT" = 90 ]; then
  export CUDA_ARCH_LIST="Kepler Maxwell Pascal 7.0+PTX";
else
  echo "CUDA version ${CUDA_VERSION_SHORT} not supported!" >&2 && exit 1;
fi;

PYTHON_VERSIONS=(python2.7 python3.5 python3.6);
PYTHON_NUMBERS=(27 35 36);
for i in $(seq ${#PYTHON_VERSIONS[@]}); do
  export PYTHON=${PYTHON_VERSIONS[i - 1]}
  export PYV=${PYTHON_NUMBERS[i - 1]};
  source "/py${PYV}-cuda/bin/activate";

  mkdir /tmp/src/build-py$PYV-cuda;
  cd /tmp/src/build-py$PYV-cuda;
  cmake -DWITH_CUDA=ON -DCUDA_ARCH_LIST="$CUDA_ARCH_LIST" \
	-DWITH_GLOG=OFF -DWITH_TESTS=OFF -DWITH_IPP=OFF \
	-DCMAKE_BUILD_TYPE=RELEASE ..;
  make;
  cd pytorch;
  python setup.py bdist_wheel;
  cp dist/*.whl /host/tmp/imgdistort/wheels/cu${CUDA_VERSION_SHORT};

  # Install imgdistort wheel.
  pip install $(find dist/ -name "*.whl");

  # Move to the tmp directory to ensure that nothing gets imported from the
  # build directory.
  cd /tmp;

  # First, ensure that CUDA is available.
  python -c \
      "import imgdistort_pytorch; assert(imgdistort_pytorch.is_cuda_available())";

  # Test installed module.
  python -m unittest imgdistort_pytorch.affine_test;
  python -m unittest imgdistort_pytorch.morphology_test;

  deactivate;
done;

CUV=cu${CUDA_VERSION_SHORT};
echo "";
echo "";
echo "=========================================================="
echo "== Python wheels located at /tmp/imgdistort/wheels/$CUV =="
echo "=========================================================="
