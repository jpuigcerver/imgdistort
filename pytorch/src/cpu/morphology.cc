#include <THW/THTensor.h>

#include <imgdistort/cpu/morphology.h>
#include <torch/morphology.h>

extern "C" {
#include <pytorch/src/cpu/morphology.h>
}

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

namespace imgdistort {
namespace pytorch {
namespace cpu {

template <typename T>
class DilateCaller : public torch::MorphologyCaller<T> {
 public:
  void operator()(
      const long N, const long C, const long H, const long W,
      const long Mn, const long* Ms, const uint8_t* M,
      const T* src, T* dst) const override {
    ::imgdistort::cpu::dilate_nchw<T, long>(
         N, C, H, W, Mn, Ms, M, src, W, dst, W);
  }
};

template <typename T>
class ErodeCaller : public torch::MorphologyCaller<T> {
 public:
  void operator()(
      const long N, const long C, const long H, const long W,
      const long Mn, const long* Ms, const uint8_t* M,
      const T* src, T* dst) const {
    ::imgdistort::cpu::erode_nchw<T, long>(
         N, C, H, W, Mn, Ms, M, src, W, dst, W);
  }
};

}  // namespace cpu
}  // namespace pytorch
}  // namespace imgdistort

#define DEFINE_WRAPPER(TSNAME, TRTYPE, TITYPE, TBTYPE)                  \
  void imgdistort_pytorch_cpu_dilate_nchw_##TSNAME(                     \
      const TITYPE* kernel_sizes, const TBTYPE* kernels,                \
      const TRTYPE* input, TRTYPE* output) {                            \
    typedef TensorTraits<TRTYPE>::DType DType;                          \
    ConstTensor<TITYPE> t_kernel_sizes(kernel_sizes);                   \
    ConstTensor<TBTYPE> t_kernels(kernels);                             \
    ConstTensor<TRTYPE> t_input(input);                                 \
    MutableTensor<TRTYPE> t_output(output);                             \
    ::imgdistort::torch::morphology_nchw<TRTYPE, TITYPE, TBTYPE>(       \
         t_kernel_sizes, t_kernels, t_input, &t_output,                 \
         ::imgdistort::pytorch::cpu::DilateCaller<DType>());            \
  }                                                                     \
                                                                        \
  void imgdistort_pytorch_cpu_erode_nchw_##TSNAME(                      \
      const TITYPE* kernel_sizes, const TBTYPE* kernels,                \
      const TRTYPE* input, TRTYPE* output) {                            \
    typedef TensorTraits<TRTYPE>::DType DType;                          \
    ConstTensor<TITYPE> t_kernel_sizes(kernel_sizes);                   \
    ConstTensor<TBTYPE> t_kernels(kernels);                             \
    ConstTensor<TRTYPE> t_input(input);                                 \
    MutableTensor<TRTYPE> t_output(output);                             \
    ::imgdistort::torch::morphology_nchw<TRTYPE, TITYPE, TBTYPE>(       \
         t_kernel_sizes, t_kernels, t_input, &t_output,                 \
           ::imgdistort::pytorch::cpu::ErodeCaller<DType>());           \
  }

DEFINE_WRAPPER(f32, THFloatTensor, THLongTensor, THByteTensor)
DEFINE_WRAPPER(f64, THDoubleTensor, THLongTensor, THByteTensor)
DEFINE_WRAPPER(s8, THCharTensor, THLongTensor, THByteTensor)
DEFINE_WRAPPER(s16, THShortTensor, THLongTensor, THByteTensor)
DEFINE_WRAPPER(s32, THIntTensor, THLongTensor, THByteTensor)
DEFINE_WRAPPER(s64, THLongTensor, THLongTensor, THByteTensor)
DEFINE_WRAPPER(u8, THByteTensor, THLongTensor, THByteTensor)
