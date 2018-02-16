#include <THW/THCTensor.h>

#include <imgdistort/gpu/affine.h>
#include <torch/affine.h>

extern "C" {
#include <pytorch/src/gpu/affine.h>
}

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

extern THCState* state;  // Defined by PyTorch

namespace imgdistort {
namespace pytorch {
namespace gpu {

template <typename T>
class AffineCaller : public torch::AffineCaller<T> {
 public:
  void operator()(
      const long N, const long C, const long H, const long W,
      const long Mn, const double* M, const T* src, T* dst,
      const T& border_value) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::imgdistort::gpu::affine_nchw<T, long>(
         N, C, H, W, Mn, M, src, W, dst, W, border_value, stream);
  }
};

}  // namespace gpu
}  // namespace pytorch
}  // namespace imgdistort

#define DEFINE_WRAPPER(TSNAME, DTYPE, TTYPE, TDTYPE)                    \
  void imgdistort_pytorch_gpu_affine_nchw_##TSNAME(                     \
      const TDTYPE* affine_matrix, const TTYPE* input, TTYPE* output,   \
      const DTYPE border_value) {                                       \
    typedef TensorTraits<TTYPE>::DType DType;                           \
    ConstTensor<TDTYPE> t_affine(affine_matrix, state);                 \
    ConstTensor<TTYPE> t_input(input, state);                           \
    MutableTensor<TTYPE> t_output(output, state);                       \
    ::imgdistort::torch::affine_nchw<TTYPE, TDTYPE>(                    \
         t_affine, t_input, &t_output, border_value,                    \
         ::imgdistort::pytorch::gpu::AffineCaller<DType>());            \
  }

DEFINE_WRAPPER(f32, float, THCudaTensor, THCudaDoubleTensor)
DEFINE_WRAPPER(f64, double, THCudaDoubleTensor, THCudaDoubleTensor)
DEFINE_WRAPPER(s8, int8_t, THCudaCharTensor, THCudaDoubleTensor)
DEFINE_WRAPPER(s16, int16_t, THCudaShortTensor, THCudaDoubleTensor)
DEFINE_WRAPPER(s32, int32_t, THCudaIntTensor, THCudaDoubleTensor)
DEFINE_WRAPPER(s64, int64_t, THCudaLongTensor, THCudaDoubleTensor)
DEFINE_WRAPPER(u8, uint8_t, THCudaByteTensor, THCudaDoubleTensor)
