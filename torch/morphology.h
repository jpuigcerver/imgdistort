#ifndef IMGDISTORT_TORCH_MORPHOLOGY_H_
#define IMGDISTORT_TORCH_MORPHOLOGY_H_

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
class MorphologyCaller {
 public:
  virtual void operator()(
      const long N, const long C, const long H, const long W,
      const long Mn, const long* Ms, const uint8_t* M,
      const T* src, T* dst) const = 0;
};

template <typename TR, typename TI, typename TB>
void morphology_nchw(
    const ConstTensor<TI>& kernel_sizes, const ConstTensor<TB>& kernels,
    const ConstTensor<TR>& input, MutableTensor<TR>* output,
    const MorphologyCaller<typename TensorTraits<TR>::DType>& caller) {
  assert(kernel_sizes.Dims() == 1 || kernel_sizes.Dims() == 2);
  assert((kernel_sizes.Dims() == 1 && kernel_sizes.Size(0) == 2) ||
         (kernel_sizes.Dims() == 2 && kernel_sizes.Size(1) == 2));
  assert(kernel_sizes.IsContiguous());
  assert(kernels.IsContiguous());
  assert(input.Dims() == 4);
  assert(input.IsContiguous());
  output->ResizeAs(input);
  assert(output->IsContiguous());

  const long N = input.Size(0);
  const long C = input.Size(1);
  const long H = input.Size(2);
  const long W = input.Size(3);
  const long Mn = kernel_sizes.Dims() == 2 ? kernel_sizes.Size(0) : 1;

  caller(N, C, H, W, Mn, kernel_sizes.Data(), kernels.Data(), input.Data(),
         output->Data());
}

}  // namespace torch
}  // namespace imgdistort

#endif  // IMGDISTORT_TORCH_MORPHOLOGY_H_
