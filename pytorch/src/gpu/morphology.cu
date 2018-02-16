#include <THW/THCTensor.h>

#include <imgdistort/gpu/morphology.h>
#include <torch/morphology.h>

extern "C" {
#include <pytorch/src/gpu/morphology.h>
}

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

extern THCState* state;  // Defined by PyTorch

namespace imgdistort {
namespace pytorch {
namespace gpu {

template <typename T>
class DilateCaller : public torch::MorphologyCaller<T> {
 public:
  void operator()(
      const long N, const long C, const long H, const long W,
      const long Mn, const long* Ms, const uint8_t* M,
      const T* src, T* dst) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::imgdistort::gpu::dilate_nchw<T, long>(
         N, C, H, W, Mn, Ms, M, src, W, dst, W, stream);
  }
};

template <typename T>
class ErodeCaller : public torch::MorphologyCaller<T> {
 public:
  void operator()(
      const long N, const long C, const long H, const long W,
      const long Mn, const long* Ms, const uint8_t* M,
      const T* src, T* dst) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::imgdistort::gpu::erode_nchw<T, long>(
         N, C, H, W, Mn, Ms, M, src, W, dst, W, stream);
  }
};

}  // namespace gpu
}  // namespace pytorch
}  // namespace imgdistort

#define DEFINE_WRAPPER(TSNAME, TRTYPE, TITYPE, TBTYPE)                  \
  void imgdistort_pytorch_gpu_dilate_nchw_##TSNAME(                     \
      const TITYPE* kernel_sizes, const TBTYPE* kernels,                \
      const TRTYPE* input, TRTYPE* output) {                            \
    typedef TensorTraits<TRTYPE>::DType DType;                          \
    ConstTensor<TITYPE> t_kernel_sizes(kernel_sizes, state);            \
    ConstTensor<TBTYPE> t_kernels(kernels, state);                      \
    ConstTensor<TRTYPE> t_input(input, state);                          \
    MutableTensor<TRTYPE> t_output(output, state);                      \
    ::imgdistort::torch::morphology_nchw<TRTYPE, TITYPE, TBTYPE>(       \
         t_kernel_sizes, t_kernels, t_input, &t_output,                 \
         ::imgdistort::pytorch::gpu::DilateCaller<DType>());            \
  }                                                                     \
                                                                        \
  void imgdistort_pytorch_gpu_erode_nchw_##TSNAME(                      \
      const TITYPE* kernel_sizes, const TBTYPE* kernels,                \
      const TRTYPE* input, TRTYPE* output) {                            \
    typedef TensorTraits<TRTYPE>::DType DType;                          \
    ConstTensor<TITYPE> t_kernel_sizes(kernel_sizes, state);            \
    ConstTensor<TBTYPE> t_kernels(kernels, state);                      \
    ConstTensor<TRTYPE> t_input(input, state);                          \
    MutableTensor<TRTYPE> t_output(output, state);                      \
    ::imgdistort::torch::morphology_nchw<TRTYPE, TITYPE, TBTYPE>(       \
         t_kernel_sizes, t_kernels, t_input, &t_output,                 \
           ::imgdistort::pytorch::gpu::ErodeCaller<DType>());           \
  }

DEFINE_WRAPPER(f32, THCudaTensor, THCudaLongTensor, THCudaByteTensor)
DEFINE_WRAPPER(f64, THCudaDoubleTensor, THCudaLongTensor, THCudaByteTensor)
DEFINE_WRAPPER(s8, THCudaCharTensor, THCudaLongTensor, THCudaByteTensor)
DEFINE_WRAPPER(s16, THCudaShortTensor, THCudaLongTensor, THCudaByteTensor)
DEFINE_WRAPPER(s32, THCudaIntTensor, THCudaLongTensor, THCudaByteTensor)
DEFINE_WRAPPER(s64, THCudaLongTensor, THCudaLongTensor, THCudaByteTensor)
DEFINE_WRAPPER(u8, THCudaByteTensor, THCudaLongTensor, THCudaByteTensor)
