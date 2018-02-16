#include <THW/THTensor.h>

#include <imgdistort/cpu/affine.h>
#include <torch/affine.h>

extern "C" {
#include <pytorch/src/cpu/affine.h>
}

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

namespace imgdistort {
namespace pytorch {
namespace cpu {

template <typename T>
class AffineCaller : public torch::AffineCaller<T> {
 public:
  void operator()(
      const long N, const long C, const long H, const long W,
      const long Mn, const double* M, const T* src, T* dst,
      const T& border_value) const override {
    ::imgdistort::cpu::affine_nchw<T, long>(
         N, C, H, W, Mn, M, src, W, dst, W, border_value);
  }
};

}  // namespace cpu
}  // namespace pytorch
}  // namespace imgdistort

#define DEFINE_WRAPPER(TSNAME, DTYPE, TTYPE, TDTYPE)                    \
  void imgdistort_pytorch_cpu_affine_nchw_##TSNAME(                     \
      const TDTYPE* affine_matrix, const TTYPE* input, TTYPE* output,   \
      const DTYPE border_value) {                                       \
    typedef TensorTraits<TTYPE>::DType DType;                           \
    ConstTensor<TDTYPE> t_affine(affine_matrix);                        \
    ConstTensor<TTYPE> t_input(input);                                  \
    MutableTensor<TTYPE> t_output(output);                              \
    ::imgdistort::torch::affine_nchw<TTYPE, TDTYPE>(                    \
         t_affine, t_input, &t_output, border_value,                    \
         ::imgdistort::pytorch::cpu::AffineCaller<DType>());            \
  }

DEFINE_WRAPPER(f32, float, THFloatTensor, THDoubleTensor)
DEFINE_WRAPPER(f64, double, THDoubleTensor, THDoubleTensor)
DEFINE_WRAPPER(s8, int8_t, THCharTensor, THDoubleTensor)
DEFINE_WRAPPER(s16, int16_t, THShortTensor, THDoubleTensor)
DEFINE_WRAPPER(s32, int32_t, THIntTensor, THDoubleTensor)
DEFINE_WRAPPER(s64, int64_t, THLongTensor, THDoubleTensor)
DEFINE_WRAPPER(u8, uint8_t, THByteTensor, THDoubleTensor)
