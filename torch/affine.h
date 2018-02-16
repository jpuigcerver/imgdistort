#ifndef IMGDISTORT_TORCH_AFFINE_H_
#define IMGDISTORT_TORCH_AFFINE_H_

#include <cassert>

namespace nnutils {
namespace THW {
template <typename THTensor> class ConstTensor;
template <typename THTensor> class MutableTensor;
template <typename THTensor> class TensorTraits;
}
}

namespace imgdistort {
namespace torch {

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

template <typename T>
class AffineCaller {
 public:
  virtual void operator()(
      const long N, const long C, const long H, const long W,
      const long Mn, const double* M, const T* src, T* dst,
      const T& border_value) const = 0;
};

template <typename T, typename TD>
void affine_nchw(
    const ConstTensor<TD>& affine_matrix, ConstTensor<T>& input,
    MutableTensor<T>* output,
    const typename TensorTraits<T>::DType& border_value,
    const AffineCaller<typename TensorTraits<T>::DType>& caller) {
  assert(affine_matrix.Dims() == 2 || affine_matrix.Dims() == 3);
  assert(affine_matrix.IsContiguous());
  assert(input.Dims() == 4);
  assert(input.IsContiguous());
  output->ResizeAs(input);
  assert(output->IsContiguous());

  const long N = input.Size(0);
  const long C = input.Size(1);
  const long H = input.Size(2);
  const long W = input.Size(3);
  const long Mn = affine_matrix.Dims() == 3 ? affine_matrix.Size(0) : 1;

  caller(N, C, H, W, Mn, affine_matrix.Data(), input.Data(), output->Data(),
         border_value);
}

}  // namespace torch
}  // namespace imgdistort

#endif  // IMGDISTORT_TORCH_AFFINE_H_
