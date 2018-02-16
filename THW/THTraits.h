#ifndef NNUTILS_THW_THTRAITS_H_
#define NNUTILS_THW_THTRAITS_H_

#include <TH/THTensor.h>

namespace nnutils {
namespace THW {

// Traits for TH's tensor types.
// TensorTraits<T>::THType -> TH tensor type, which equals the the template T.
// TensorTraits<T>::DType  -> Data type (char, int, float, etc).
template <typename T> class TensorTraits;

}  // namespace nnutils
}  // namespace THW

#include <THW/generic/THTraits.h>
#include <TH/THGenerateAllTypes.h>

#endif  // NNUTILS_THW_THTRAITS_H_
