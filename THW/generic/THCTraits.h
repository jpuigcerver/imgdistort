#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THW/generic/THCTraits.h"
#else

namespace nnutils {
namespace THW {

template <>
class TensorTraits<THCTensor> {
 public:
  typedef THCTensor TType;
  typedef real      DType;
  typedef real      VType;
};

template <>
class TensorToCpuTraits<THCTensor> {
 public:
  typedef THTensor Type;
};

template <>
class TensorToGpuTraits<THTensor> {
 public:
  typedef THCTensor Type;
};


}  // namespace nnutils
}  // namespace THW

#endif
