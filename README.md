# imgdistort
Library to perform image distortions on the GPU and CPU.

## Available distortions
- [Affine transformations](https://en.wikipedia.org/wiki/Affine_transformation)
  + Scaling, translation, rotation, shearing, etc.
- Morphological operations
  + [Grayscale erosion](https://en.wikipedia.org/wiki/Erosion_(morphology))
  + [Grayscale dilation](https://en.wikipedia.org/wiki/Dilation_(morphology))
  
## Requirements
- A modern C++ compiler with C++11 support (e.g. GCC >= 4.9). 
  You only need a C++11 compiler to build the library, once it is built you 
  can link it as a C library. It is also recommended that the compiler offers 
  OpenMP support, but it is not required. 
- [CMake](https://cmake.org/) >= 3.0
- [Google Glog](https://github.com/google/glog)
- Recommended: [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) >= 6.5.
  Only necessary if you want to add GPU support. 
- Recommended: [Intel® Integrated Performance Primitives](https://software.intel.com/en-us/intel-ipp)
  It's recommended to use Intel's implementation of the affine and 
  morphological distortions. If you don't have a license for it, a custom 
  implementation will be used. 
- Optional: [Google Test suite](https://github.com/google/googletest) 
  (including both Google Test and Google Mock), only necessary if yo want to 
  build the tests.
  
## Installation 
You can build and install the library using CMake, just clone the repository, 
create a build directory and build it:

```bash
git clone https://github.com/jpuigcerver/imgdistort
cd imgdistort
mkdir build
cd build
ccmake ..
make
make install
```

CMake should detect the libraries that you have installed and automatically 
configure them. If you have some of the required/recommended libraries 
installed in a non-standard location, you can make use of some CMake variables 
to specify the location:

```bash
cmake                                   \
  -DIPP_ROOT_DIR=/path/to/ipp           \
  -DGLOG_ROOT_DIR=/path/to/glog         \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda \
  ..
```

If CMake detects some library that you don't want to use, you can force to 
ignore them:
```bash
cmake -DWITH_IPP=OFF -DWITH_CUDA=OFF ..
```

To build the tests, CMake will need to find Google Test and Google Mock, 
and you will need to enable the test build:
```bash
cmake -DGTEST_ROOT=/path/to/gtest -DGMOCK_ROOT=/path/to/gmock -DWITH_TESTS=ON ..
```

## Image format
Pixel intensities can be represented by different data types:
- unsigned 8-bit integer (uint8_t)
- unsigned 16-bit integer (uint16_t)
- unsigned 32-bit integer (uint32_t)
- unsigned 64-bit integer (uint64_t)
- signed 16-bit integer (int16_t)
- signed 32-bit integer (int32_t)
- signed 64-bit integer (int64_t)
- single precision floating numbers (float)
- double precision floating numbers (double)

The library is designed to work with batched images, i.e. processing multiple
images simultaneously. The layout for each batch of images is:
Batch size x Channels x Height x Width (which is the standard layout used 
in [Torch](http://torch.ch/)).

It is important to keep in mind that the output images have the same size as 
the original images, regardless of the applied operation. That means that you 
may "lose" part of your input image when certain transformations are applied 
(i.e. affine transformations). To avoid that, pad your images conveniently
before using imgdistort.

Additionally, all batched images must have the same size, so if your images 
have different sizes you will also need to pad them.
