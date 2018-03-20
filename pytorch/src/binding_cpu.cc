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
namespace cpu {

template <typename T>
class AffineCaller : public ::imgdistort::pytorch::AffineCaller<T> {
 public:
  void operator()(
      const int N, const int C, const int H, const int W,
      const double* M, const int Mn, const T* src, T* dst,
      const T& border_value) const override {
    ::imgdistort::cpu::affine_nchw<T>(N, C, H, W, M, Mn, src, W, dst, W,
                                      border_value);
  };
};

template <typename T>
class DilateCaller : public ::imgdistort::pytorch::MorphologyCaller<T> {
 public:
  void operator()(
      const int N, const int C, const int H, const int W,
      const uint8_t* M, const int* Ms, const int Mn,
      const T* src, T* dst) const {
    ::imgdistort::cpu::dilate_nchw<T>(N, C, H, W, M, Ms, Mn, src, W, dst, W);
  }
};

template <typename T>
class ErodeCaller : public ::imgdistort::pytorch::MorphologyCaller<T> {
 public:
  void operator()(
      const int N, const int C, const int H, const int W,
      const uint8_t* M, const int* Ms, const int Mn,
      const T* src, T* dst) const {
    ::imgdistort::cpu::erode_nchw<T>(N, C, H, W, M, Ms, Mn, src, W, dst, W);
  }
};

}  // namespace cpu
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
