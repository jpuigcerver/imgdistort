#ifndef NNUTILS_THW_THCTENSORTEST_H_
#define NNUTILS_THW_THCTENSORTEST_H_

#include <THW/THTensorTest.h>
#include <THW/THCTraits.h>

namespace nnutils {
namespace THW {
namespace testing {

static THCState* kState = nullptr;

template <typename THTensor>
typename TensorToCpuTraits<THTensor>::Type*
THTensor_clone2cpu(THTensor* tensor);

template <typename THTensor>
typename TensorToGpuTraits<THTensor>::Type*
THTensor_clone2gpu(THTensor* tensor);

}  // namespace testing
}  // namespace THW
}  // namespace nnutils

#include <THW/generic/THCTensorTest.h>
#include <THC/THCGenerateAllTypes.h>

#endif  // NNUTILS_THW_THCTENSORTEST_H_
