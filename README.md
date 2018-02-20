# imgdistort

[![Build Status](https://travis-ci.org/jpuigcerver/imgdistort.svg?branch=npp)](https://travis-ci.org/jpuigcerver/imgdistort)

A library to perform image distortions on the CPU and GPU (CUDA).

## Available distortions
- [Affine transformations](https://en.wikipedia.org/wiki/Affine_transformation)
- [Grayscale dilation](https://en.wikipedia.org/wiki/Dilation_(morphology))
- [Grayscale erosion](https://en.wikipedia.org/wiki/Erosion_(morphology))
  
## Requirements

### Minimum:
- C++11 compiler (tested with GCC 4.8.2, 5.4.0, Clang 3.5.0).
- [CMake 3.0](https://cmake.org/).

### Recommended:
- For better logging: [Google Glog](https://github.com/google/glog).
- For GPU support: [CUDA Toolkit](https://developer.nvidia.com/cuda-zone).
- For running tests: [Google Test](https://github.com/google/googletest).

### PyTorch bindings:
- [PyTorch](http://pytorch.org/) (tested with version 0.3.0).
  
## Installation

### PyTorch bindings with pip

The easiest way of using imgdistort with PyTorch is using pip. I have 
precompiled the tool for Linux using different version of Python
and supporting different devices. The value in each cell corresponds to 
the commit from which the wheel was built.

|          | Python 2.7 | Python 3.5 | Python 3.6 |
|----------|:----------:|:----------:|:----------:|
| CPU-only | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cpu/imgdistort_pytorch-0.1.0+e5fa06d-cp27-cp27mu-linux_x86_64.whl) | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cpu/imgdistort_pytorch-0.1.0+e5fa06d-cp35-cp35m-linux_x86_64.whl) | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cpu/imgdistort_pytorch-0.1.0+e5fa06d-cp36-cp36m-linux_x86_64.whl) |
| CUDA 7.5 | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cu75/imgdistort_pytorch_cu75-0.1.0+e5fa06d-cp27-cp27mu-linux_x86_64.whl) | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cu75/imgdistort_pytorch_cu75-0.1.0+e5fa06d-cp35-cp35m-linux_x86_64.whl) | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cu75/imgdistort_pytorch_cu75-0.1.0+e5fa06d-cp36-cp36m-linux_x86_64.whl) |
| CUDA 8.0 | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cu80/imgdistort_pytorch_cu80-0.1.0+e5fa06d-cp27-cp27mu-linux_x86_64.whl) | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cu80/imgdistort_pytorch_cu80-0.1.0+e5fa06d-cp35-cp35m-linux_x86_64.whl) | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cu80/imgdistort_pytorch_cu80-0.1.0+e5fa06d-cp36-cp36m-linux_x86_64.whl) |
| CUDA 9.0 | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cu90/imgdistort_pytorch_cu90-0.1.0+e5fa06d-cp27-cp27mu-linux_x86_64.whl) | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cu90/imgdistort_pytorch_cu90-0.1.0+e5fa06d-cp35-cp35m-linux_x86_64.whl) | [e5fa06d](https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cu90/imgdistort_pytorch_cu90-0.1.0+e5fa06d-cp36-cp36m-linux_x86_64.whl) |

For instance, to install the CPU-only version for Python 3.5:
```bash
pip3 install https://www.prhlt.upv.es/~jpuigcerver/imgdistort/whl/cpu/imgdistort_pytorch-0.1.0+e5fa06d-cp35-cp35m-linux_x86_64.whl
```

Notice that each version of the library was compiled to support only the most
common and supported architectures in each CUDA release. 
Choose the compiled version accordingly:

|          | Supported architectures        | Compute Capability                |
|----------|-------------------------------:|----------------------------------:|
| CUDA 7.5 | Kepler, Maxwell                | 3.0, 3.5, 5.0, 5.2                |
| CUDA 8.0 | Kepler, Maxwell, Pascal        | 3.0, 3.5, 5.0, 5.2, 6.0, 6.1      |
| CUDA 9.0 | Kepler, Maxwell, Pascal, Volta | 3.0, 3.5, 5.0, 5.2, 6.0, 6.1, 7.0 |

### From sources

The installation process should be pretty straightforward assuming that you
have installed correctly the required libraries and tools.

```bash
git clone https://github.com/jpuigcerver/imgdistort.git
cd imgdistort
mkdir build
cd build
cmake ..
make
make install
```

By default, it will try to compile the PyTorch bindings with CUDA support and
install them in the default location for Python libraries in your system.

If you have any problem installing the library, read through the CMake errors
and warnings. In most cases, the problems are due to installing the tools in
non-standard locations or using old versions of them.

You can set many CMake variables to aid it to detect the required software.
Some helpful variables are:

- `CUDA_TOOLKIT_ROOT_DIR`: Specify the directory where you installed the
  NVIDIA CUDA Toolkit.
- `CUDA_ARCH_LIST`: Specify the list of CUDA architectures that should be
  supported during the compilation. By default it will use "Auto", which will
  compile _only_ for the architectures supported by your graphic cards.
- `Python_ADDITIONAL_VERSIONS`: When you have multiple versions of Python
  installed in your system, you can choose to use a specific one (e.g. 3.5)
  with this variable.
- `PYTORCH_SETUP_PREFIX`: Prefix location to install the PyTorch bindings
  (e.g. /home/jpuigcerver/.local).
  
