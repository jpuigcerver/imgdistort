#ifndef IMGDISTORT_CPU_MORPHOLOGY_H_
#define IMGDISTORT_CPU_MORPHOLOGY_H_

#include <imgdistort/config.h>
#include <imgdistort/cpu/morphology_generic.h>

#ifdef WITH_IPP
#include <imgdistort/cpu/morphology_ipp.h>
#endif // WITH_IPP

#include <cstdint>
#include <vector>

#ifdef __cplusplus
namespace imgdistort {
namespace cpu {

//
//
// @param N number of images in the batch
// @param C number of channels per image
// @param H height of the batch (maximum height among all images)
// @param W width of the batch (maximum width among all images)
// @param Mn number of structuring element matrices
// @param Ms size of each structuring element matrix (Mn x 2 array)
// @param M structuring element matrices (Mn x ? x ? array)
// @param src source batch
// @param sp source pitch (or stride)
// @param dst destination
// @param dp destination pitch (or stride)
template <typename T, typename Int>
inline void dilate_nchw(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const Int* Ms, const uint8_t* M,
    const T* src, const Int sp, T* dst, const Int dp) {
#ifdef WITH_IPP
  if (sizeof(Int) == sizeof(int)) {
    dilate_nchw_ipp<T>(
        N, C, H, W, Mn, reinterpret_cast<const int*>(Ms), M, src, sp, dst, dp);
  } else {
    std::vector<int> Ms_(Ms, Ms + Mn);
    dilate_nchw_ipp<T>(N, C, H, W, Mn, Ms_.data(), M, src, sp, dst, dp);
  }
#else
  dilate_nchw_generic<T, Int>(N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
#endif
}

//
//
// @param N number of images in the batch
// @param C number of channels per image
// @param H height of the batch (maximum height among all images)
// @param W width of the batch (maximum width among all images)
// @param Mn number of structuring element matrices
// @param Ms size of each structuring element matrix (Mn x 2 array)
// @param M structuring element matrices (Mn x ? x ? array)
// @param src source batch
// @param sp source pitch (or stride)
// @param dst destination
// @param dp destination pitch (or stride)
template <typename T, typename Int>
inline void erode_nchw(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const Int* Ms, const uint8_t* M,
    const T* src, const Int sp, T* dst, const Int dp) {
#ifdef WITH_IPP
  if (sizeof(Int) == sizeof(int)) {
    dilate_nchw_ipp<T>(
        N, C, H, W, Mn, reinterpret_cast<const int*>(Ms), M, src, sp, dst, dp);
  } else {
    std::vector<int> Ms_(Ms, Ms + Mn);
    erode_nchw_ipp<T>(N, C, H, W, Mn, Ms_.data(), M, src, sp, dst, dp);
  }
#else
  erode_nchw_generic<T, Int>(N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
#endif
}

}  // namespace cpu
}  // namespace imgdistort
#endif  // __cplusplus

// C bindings
#define DECLARE_BINDING(TYPE, SNAME)                              \
  EXTERN_C void imgdistort_cpu_dilate_nchw_##SNAME(               \
      const int N, const int C, const int H, const int W,         \
      const int Mn, const int* Ms, const uint8_t* M,              \
      const TYPE* src, const int sp, TYPE* dst, const int dp);    \
                                                                  \
  EXTERN_C void imgdistort_cpu_erode_nchw_##SNAME(                \
      const int N, const int C, const int H, const int W,         \
      const int Mn, const int* Ms, const uint8_t* M,              \
      const TYPE* src, const int sp, TYPE* dst, const int dp)

DECLARE_BINDING(float, f32);
DECLARE_BINDING(double, f64);
DECLARE_BINDING(int8_t, s8);
DECLARE_BINDING(int16_t, s16);
DECLARE_BINDING(int32_t, s32);
DECLARE_BINDING(int64_t, s64);
DECLARE_BINDING(uint8_t, u8);
DECLARE_BINDING(uint16_t, u16);
DECLARE_BINDING(uint32_t, u32);
DECLARE_BINDING(uint64_t, u64);
#undef DECLARE_BINDING

#endif  // IMGDISTORT_MORPHOLOGY_CPU_H_
