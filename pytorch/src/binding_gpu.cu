#include <THC.h>
#include <THCTensor.h>
#include <pytorch/src/binding_common.h>
#include <imgdistort/affine_gpu.h>
#include <imgdistort/morphology_gpu.h>

extern "C" {
#include <pytorch/src/binding_gpu.h>
}

#include <cassert>
#include <cstdint>

extern THCState* state;

namespace imgdistort {
namespace pytorch {

template <typename T>
inline void wrap_affine_call(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn, const T* src, T* dst) {
  cudaStream_t stream = THCState_getCurrentStream(state);
  imgdistort::gpu::affine_nchw(N, C, H, W, M, Mn, src, 0, dst, 0, stream);
}

template <MorphOp op, typename DTYPE>
inline void wrap_morph_call(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const DTYPE* src, DTYPE* dst) {
  cudaStream_t stream = THCState_getCurrentStream(state);
  if (op == DILATE) {
    imgdistort::gpu::dilate_nchw<DTYPE>(N, C, H, W, M, Ms, Mn, src, 0, dst, 0,
                                        stream);
  } else {
    imgdistort::gpu::erode_nchw<DTYPE>(N, C, H, W, M, Ms, Mn, src, 0, dst, 0,
                                       stream);
  }
}

}  // namespace pytorch
}  // namespace imgdistort

// AFFINE operations
DEFINE_AFFINE_WRAPPER(gpu, f32, float, THCudaDoubleTensor, THCudaTensor)
DEFINE_AFFINE_WRAPPER(gpu, f64, double, THCudaDoubleTensor, THCudaDoubleTensor)

DEFINE_AFFINE_WRAPPER(gpu, s8,  int8_t,  THCudaDoubleTensor, THCudaCharTensor)
DEFINE_AFFINE_WRAPPER(gpu, s16, int16_t, THCudaDoubleTensor, THCudaShortTensor)
DEFINE_AFFINE_WRAPPER(gpu, s32, int32_t, THCudaDoubleTensor, THCudaIntTensor)
DEFINE_AFFINE_WRAPPER(gpu, s64, int64_t, THCudaDoubleTensor, THCudaLongTensor)
DEFINE_AFFINE_WRAPPER(gpu, u8, uint8_t, THCudaDoubleTensor, THCudaByteTensor)

// MORPHOLOGY operations
DEFINE_MORPHOLOGY_WRAPPER(gpu, f32, float,   THCudaByteTensor, THCudaTensor)
DEFINE_MORPHOLOGY_WRAPPER(gpu, f64, double,  THCudaByteTensor,
                          THCudaDoubleTensor)

DEFINE_MORPHOLOGY_WRAPPER(gpu, s8,  int8_t,  THCudaByteTensor, THCudaCharTensor)
DEFINE_MORPHOLOGY_WRAPPER(gpu, s16, int16_t, THCudaByteTensor, THCudaShortTensor)
DEFINE_MORPHOLOGY_WRAPPER(gpu, s32, int32_t, THCudaByteTensor, THCudaIntTensor)
DEFINE_MORPHOLOGY_WRAPPER(gpu, s64, int64_t, THCudaByteTensor, THCudaLongTensor)
DEFINE_MORPHOLOGY_WRAPPER(gpu, u8,  uint8_t, THCudaByteTensor, THCudaByteTensor)
