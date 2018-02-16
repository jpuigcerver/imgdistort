#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THW/generic/THCTensorTest.h"
#else

namespace nnutils {
namespace THW {
namespace testing {

template <>
THCTensor* THTensor_new<THCTensor>(void) {
  return THCTensor_(new)(kState);
}

template <>
THCTensor* THTensor_newWithSize2d<THCTensor>(int s1, int s2) {
  return THCTensor_(newWithSize2d)(kState, s1, s2);
}

template <>
void THTensor_free<THCTensor>(THCTensor* tensor) {
  THCTensor_(free)(kState, tensor);
}


// TODO(jpuigcerver): Half support is not ready.
#if !defined(THC_REAL_IS_HALF)

template <>
typename TensorToCpuTraits<THCTensor>::Type* THTensor_clone2cpu<THCTensor>(
    THCTensor* tensor) {
  typedef TensorToCpuTraits<THCTensor>::Type CpuType;
  CpuType* t_cpu = THTensor_new<CpuType>();
  THTensor_(resizeNd)(t_cpu, tensor->nDimension, tensor->size, tensor->stride);
  THTensor_(copyCuda)(kState, t_cpu, tensor);
  return t_cpu;
}

template <>
typename TensorToGpuTraits<THTensor>::Type* THTensor_clone2gpu<THTensor>(
    THTensor* tensor) {
  typedef TensorToGpuTraits<THTensor>::Type GpuType;
  GpuType* t_gpu = THTensor_new<GpuType>();
  THCTensor_(resizeNd)(kState, t_gpu, tensor->nDimension, tensor->size,
                       tensor->stride);
  THCTensor_(copyCPU)(kState, t_gpu, tensor);
  return t_gpu;
}

#endif  // THC_REAL_IS_HALF

}  // namespace testing
}  // namespace THW
}  // namespace nnutils

#endif
