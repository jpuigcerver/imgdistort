#include <TH.h>
#include <THTensor.h>
#include <pytorch/src/binding_common.h>
#include <imgdistort/affine_cpu.h>
#include <imgdistort/morphology_cpu.h>

extern "C" {
#include <pytorch/src/binding_cpu.h>
}

#include <cassert>
#include <cstdint>

namespace imgdistort {
namespace pytorch {

template <typename T>
inline void wrap_affine_call(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn, const T* src, T* dst) {
  imgdistort::cpu::affine_nchw<T>(N, C, H, W, M, Mn, src, 0, dst, 0);
}

template <MorphOp op, typename DTYPE>
inline void wrap_morph_call(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const DTYPE* src, DTYPE* dst) {
  if (op == DILATE) {
    imgdistort::cpu::dilate_nchw<DTYPE>(N, C, H, W, M, Ms, Mn, src, 0, dst, 0);
  } else {
    imgdistort::cpu::erode_nchw<DTYPE>(N, C, H, W, M, Ms, Mn, src, 0, dst, 0);
  }
}

}  // namespace pytorch
}  // namespace imgdistort

// AFFINE operations
DEFINE_AFFINE_WRAPPER(cpu, f32, float,   THDoubleTensor, THFloatTensor)
DEFINE_AFFINE_WRAPPER(cpu, f64, double,  THDoubleTensor, THDoubleTensor)
DEFINE_AFFINE_WRAPPER(cpu, s8,  int8_t,  THDoubleTensor, THCharTensor)
DEFINE_AFFINE_WRAPPER(cpu, s16, int16_t, THDoubleTensor, THShortTensor)
DEFINE_AFFINE_WRAPPER(cpu, s32, int32_t, THDoubleTensor, THIntTensor)
DEFINE_AFFINE_WRAPPER(cpu, s64, int64_t, THDoubleTensor, THLongTensor)
DEFINE_AFFINE_WRAPPER(cpu, u8,  uint8_t, THDoubleTensor, THByteTensor)

// MORPHOLOGY operations
DEFINE_MORPHOLOGY_WRAPPER(cpu, f32, float,   THByteTensor, THFloatTensor)
DEFINE_MORPHOLOGY_WRAPPER(cpu, f64, double,  THByteTensor, THDoubleTensor)
DEFINE_MORPHOLOGY_WRAPPER(cpu, s8,  int8_t,  THByteTensor, THCharTensor)
DEFINE_MORPHOLOGY_WRAPPER(cpu, s16, int16_t, THByteTensor, THShortTensor)
DEFINE_MORPHOLOGY_WRAPPER(cpu, s32, int32_t, THByteTensor, THIntTensor)
DEFINE_MORPHOLOGY_WRAPPER(cpu, s64, int64_t, THByteTensor, THLongTensor)
DEFINE_MORPHOLOGY_WRAPPER(cpu, u8,  uint8_t, THByteTensor, THByteTensor)
