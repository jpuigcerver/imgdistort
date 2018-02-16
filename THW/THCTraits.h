#ifndef NNUTILS_THW_THCTRAITS_H_
#define NNUTILS_THW_THCTRAITS_H_

#include <THW/THTraits.h>
#include <THC/THCTensor.h>

namespace nnutils {
namespace THW {

template <typename THTensor> class TensorToCpuTraits;

template <typename THTensor> class TensorToGpuTraits;

}  // namespace THW
}  // namespace nnutils

#include <THW/generic/THCTraits.h>
#include <THC/THCGenerateAllTypes.h>

#endif  // NNUTILS_THW_THCTRAITS_H_
