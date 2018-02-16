#ifndef IMGDISTORT_CPU_AFFINE_H_
#define IMGDISTORT_CPU_AFFINE_H_

#include <imgdistort/config.h>
#include <imgdistort/cpu/affine_generic.h>

#ifdef WITH_IPP
#include <imgdistort/cpu/affine_ipp.h>
#endif // WITH_IPP

#include <cstdint>

#ifdef __cplusplus
namespace imgdistort {
namespace cpu {

//
//
// @param N number of images in the batch
// @param C number of channels per image
// @param H height of the batch (maximum height among all images)
// @param W width of the batch (maximum width among all images)
// @param Mn number of affine matrices
// @param M matrices of the affine transformation (Mn x 2 x 3 array)
// @param src source batch
// @param sp source pitch (or stride)
// @param dst destination
// @param dp destination pitch (or stride)
template <typename T, typename Int>
inline void affine_nchw(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const double* M,
    const T* src, const Int sp, T* dst, const Int dp,
    const T& border_value) {
#ifdef WITH_IPP
  affine_nchw_ipp<T>(
      N, C, H, W, Mn, M, src, sp, dst, dp, border_value);
#else
  affine_nchw_generic<T, Int>(
      N, C, H, W, Mn, M, src, sp, dst, dp, border_value);
#endif
}

}  // namespace cpu
}  // namespace imgdistort
#endif  // __cplusplus

// C bindings
#define DECLARE_BINDING(TYPE, SNAME)                              \
  EXTERN_C void imgdistort_cpu_affine_nchw_##SNAME(               \
      const int N, const int C, const int H, const int W,         \
      const int Mn, const double* M,                              \
      const TYPE* src, const int sp, TYPE* dst, const int dp,     \
      const TYPE border_value)

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

#endif  // IMGDISTORT_CPU_AFFINE_H_
