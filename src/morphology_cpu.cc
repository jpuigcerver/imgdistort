#include <imgdistort/morphology_cpu.h>

#include <imgdistort/morphology_cpu_generic.h>

#ifdef WITH_IPP
#include <imgdistort/morphology_cpu_ipp.h>
#endif

namespace imgdistort {
namespace cpu {

#define DEFINE_GENERIC_IMPLEMENTATION(T)                                \
  template <>                                                           \
  void dilate_nchw<T>(                                                  \
      const int N, const int C, const int H, const int W,               \
      const uint8_t* M, const int* Ms, const int Mn,                    \
      const T* src, const int sp, T* dst, const int dp) {               \
    dilate_nchw_generic<T>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);    \
  }                                                                     \
  template <>                                                           \
  void erode_nchw<T>(                                                   \
      const int N, const int C, const int H, const int W,               \
      const uint8_t* M, const int* Ms, const int Mn,                    \
      const T* src, const int sp, T* dst, const int dp) {               \
    erode_nchw_generic<T>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);     \
  }

#ifdef WITH_IPP
#define DEFINE_IPP_IMPLEMENTATION(T)                                    \
  template <>                                                           \
  void dilate_nchw<T>(                                                  \
      const int N, const int C, const int H, const int W,               \
      const uint8_t* M, const int* Ms, const int Mn,                    \
      const T* src, const int sp, T* dst, const int dp) {               \
    dilate_nchw_ipp<T>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);        \
  }                                                                     \
  template <>                                                           \
  void erode_nchw<T>(                                                   \
      const int N, const int C, const int H, const int W,               \
      const uint8_t* M, const int* Ms, const int Mn,                    \
      const T* src, const int sp, T* dst, const int dp) {               \
    erode_nchw_ipp<T>(N, C, H, W, M, Ms, Mn, src, sp, dst, dp);         \
  }
#else
#define DEFINE_IPP_IMPLEMENTATION(T) DEFINE_GENERIC_IMPLEMENTATION(T)
#endif  // WITH_IPP

DEFINE_GENERIC_IMPLEMENTATION(int8_t)
DEFINE_IPP_IMPLEMENTATION(int16_t)
DEFINE_GENERIC_IMPLEMENTATION(int32_t)
DEFINE_GENERIC_IMPLEMENTATION(int64_t)
DEFINE_IPP_IMPLEMENTATION(uint8_t)
DEFINE_IPP_IMPLEMENTATION(uint16_t)
DEFINE_GENERIC_IMPLEMENTATION(uint32_t)
DEFINE_GENERIC_IMPLEMENTATION(uint64_t)
DEFINE_IPP_IMPLEMENTATION(float)
DEFINE_GENERIC_IMPLEMENTATION(double)

}  // namespace cpu
}  // namespace imgdistort
