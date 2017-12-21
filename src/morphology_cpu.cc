#include <imgdistort/morphology_cpu.h>

#include <imgdistort/morphology_cpu_generic.h>

#ifdef WITH_IPP
#include <imgdistort/morphology_cpu_ipp.h>
#endif

namespace imgdistort {
namespace cpu {

template <>
void dilate_nchw<int8_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int8_t* src, const int sp, int8_t* dst, const int dp) {
  dilate_nchw_generic<int8_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

template <>
void dilate_nchw<int16_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int16_t* src, const int sp, int16_t* dst, const int dp) {
#ifdef WITH_IPP
  dilate_nchw_ipp<int16_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#else
  dilate_nchw_generic<int16_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#endif
}

template <>
void dilate_nchw<int32_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int32_t* src, const int sp, int32_t* dst, const int dp) {
  dilate_nchw_generic<int32_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

template <>
void dilate_nchw<int64_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int64_t* src, const int sp, int64_t* dst, const int dp) {
  dilate_nchw_generic<int64_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

template <>
void dilate_nchw<uint8_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint8_t* src, const int sp, uint8_t* dst, const int dp) {
#ifdef WITH_IPP
  dilate_nchw_ipp<uint8_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#else
  dilate_nchw_generic<uint8_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#endif
}

template <>
void dilate_nchw<uint16_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint16_t* src, const int sp, uint16_t* dst, const int dp) {
#ifdef WITH_IPP
  dilate_nchw_ipp<uint16_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#else
  dilate_nchw_generic<uint16_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#endif
}

template <>
void dilate_nchw<uint32_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint32_t* src, const int sp, uint32_t* dst, const int dp) {
  dilate_nchw_generic<uint32_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

template <>
void dilate_nchw<uint64_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint64_t* src, const int sp, uint64_t* dst, const int dp) {
  dilate_nchw_generic<uint64_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

template <>
void dilate_nchw<float>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const float* src, const int sp, float* dst, const int dp) {
#ifdef WITH_IPP
  dilate_nchw_ipp<float>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#else
  dilate_nchw_generic<float>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#endif
}

template <>
void dilate_nchw<double>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const double* src, const int sp, double* dst, const int dp) {
  dilate_nchw_generic<double>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

template <>
void erode_nchw<int8_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int8_t* src, const int sp, int8_t* dst, const int dp) {
  erode_nchw_generic<int8_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

template <>
void erode_nchw<int16_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int16_t* src, const int sp, int16_t* dst, const int dp) {
#ifdef WITH_IPP
  erode_nchw_ipp<int16_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#else
  erode_nchw_generic<int16_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#endif
}

template <>
void erode_nchw<int32_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int32_t* src, const int sp, int32_t* dst, const int dp) {
  erode_nchw_generic<int32_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

template <>
void erode_nchw<int64_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int64_t* src, const int sp, int64_t* dst, const int dp) {
  erode_nchw_generic<int64_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

template <>
void erode_nchw<uint8_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint8_t* src, const int sp, uint8_t* dst, const int dp) {
#ifdef WITH_IPP
  erode_nchw_ipp<uint8_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#else
  erode_nchw_generic<uint8_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#endif
}

template <>
void erode_nchw<uint16_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint16_t* src, const int sp, uint16_t* dst, const int dp) {
#ifdef WITH_IPP
  erode_nchw_ipp<uint16_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#else
  erode_nchw_generic<uint16_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#endif
}

template <>
void erode_nchw<uint32_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint32_t* src, const int sp, uint32_t* dst, const int dp) {
  erode_nchw_generic<uint32_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

template <>
void erode_nchw<uint64_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint64_t* src, const int sp, uint64_t* dst, const int dp) {
  erode_nchw_generic<uint64_t>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

template <>
void erode_nchw<float>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const float* src, const int sp, float* dst, const int dp) {
#ifdef WITH_IPP
  erode_nchw_ipp<float>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#else
  erode_nchw_generic<float>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
#endif
}

template <>
void erode_nchw<double>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const double* src, const int sp, double* dst, const int dp) {
  erode_nchw_generic<double>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);
}

}  // namespace cpu
}  // namespace imgdistort

#define DEFINE_C_FUNCTION(DESC, TYPE)                               \
  extern "C" void imgdistort_cpu_dilate_nchw_ ## DESC  (            \
      const int N, const int C, const int H, const int W,           \
      const uint8_t* M, const int* Ms, const int Mn,                \
      const TYPE* src, const int sp, TYPE* dst, const int dp) {     \
    imgdistort::cpu::dilate_nchw<TYPE>(                             \
        N, C, H, W, M, Ms, Mn, src, sp, dst, dp);                   \
  }                                                                 \
  extern "C" void imgdistort_cpu_erode_nchw_ ## DESC  (             \
      const int N, const int C, const int H, const int W,           \
      const uint8_t* M, const int* Ms, const int Mn,                \
      const TYPE* src, const int sp, TYPE* dst, const int dp) {     \
    imgdistort::cpu::erode_nchw<TYPE>(                              \
        N, C, H, W, M, Ms, Mn, src, sp, dst, dp);                   \
  }

DEFINE_C_FUNCTION(s8,  int8_t)
DEFINE_C_FUNCTION(s16, int16_t)
DEFINE_C_FUNCTION(s32, int32_t)
DEFINE_C_FUNCTION(s64, int64_t)
DEFINE_C_FUNCTION(u8,  uint8_t)
DEFINE_C_FUNCTION(u16, uint16_t)
DEFINE_C_FUNCTION(u32, uint32_t)
DEFINE_C_FUNCTION(u64, uint64_t)
DEFINE_C_FUNCTION(f32, float)
DEFINE_C_FUNCTION(f64, double)
