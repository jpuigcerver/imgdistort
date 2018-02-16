#ifndef IMGDISTORT_CPU_AFFINE_IPP_H_
#define IMGDISTORT_CPU_AFFINE_IPP_H_

#include <imgdistort/cpu/affine_generic.h>
#include <imgdistort/logging.h>
#include <ippdefs.h>
#include <ippi.h>
#include <ipps.h>

#include <cstdint>
#include <vector>

namespace imgdistort {
namespace cpu {

namespace internal {

template <typename T>
struct IppTraits;

template <>
struct IppTraits<double> {
  static constexpr IppDataType data_type = ipp64f;
};

template <>
struct IppTraits<float> {
  static constexpr IppDataType data_type = ipp32f;
};

template <>
struct IppTraits<uint8_t> {
  static constexpr IppDataType data_type = ipp8u;
};

template <>
struct IppTraits<uint16_t> {
  static constexpr IppDataType data_type = ipp16u;
};

template <>
struct IppTraits<int16_t> {
  static constexpr IppDataType data_type = ipp16s;
};

template <typename T>
inline IppStatus wrap_ippiWarpAffineLinear_C1R(
    const T* pSrc, Ipp32s srcStep, T* pDst, Ipp32s dstStep,
    IppiPoint dstRoiOffset, IppiSize dstRoiSize, IppiWarpSpec* pSpec,
    Ipp8u* pBuffer);

template <>
inline IppStatus wrap_ippiWarpAffineLinear_C1R<double>(
    const double* pSrc, Ipp32s srcStep, double* pDst, Ipp32s dstStep,
    IppiPoint dstRoiOffset, IppiSize dstRoiSize, IppiWarpSpec* pSpec,
    Ipp8u* pBuffer) {
  return ippiWarpAffineLinear_64f_C1R(
      pSrc, srcStep, pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiWarpAffineLinear_C1R<float>(
    const float* pSrc, Ipp32s srcStep, float* pDst, Ipp32s dstStep,
    IppiPoint dstRoiOffset, IppiSize dstRoiSize, IppiWarpSpec* pSpec,
    Ipp8u* pBuffer) {
  return ippiWarpAffineLinear_32f_C1R(
      pSrc, srcStep, pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiWarpAffineLinear_C1R<int16_t>(
    const int16_t* pSrc, Ipp32s srcStep, int16_t* pDst, Ipp32s dstStep,
    IppiPoint dstRoiOffset, IppiSize dstRoiSize, IppiWarpSpec* pSpec,
    Ipp8u* pBuffer) {
  return ippiWarpAffineLinear_16s_C1R(
      pSrc, srcStep, pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiWarpAffineLinear_C1R<uint8_t>(
    const uint8_t* pSrc, Ipp32s srcStep, uint8_t* pDst, Ipp32s dstStep,
    IppiPoint dstRoiOffset, IppiSize dstRoiSize, IppiWarpSpec* pSpec,
    Ipp8u* pBuffer) {
  return ippiWarpAffineLinear_8u_C1R(
      pSrc, srcStep, pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiWarpAffineLinear_C1R<uint16_t>(
    const uint16_t* pSrc, Ipp32s srcStep, uint16_t* pDst, Ipp32s dstStep,
    IppiPoint dstRoiOffset, IppiSize dstRoiSize, IppiWarpSpec* pSpec,
    Ipp8u* pBuffer) {
  return ippiWarpAffineLinear_16u_C1R(
      pSrc, srcStep, pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec, pBuffer);
}

template <typename T>
void affine_nchw_ipp(
    const int N, const int C, const int H, const int W,
    const int Mn, const double* M,
    const T* src, const int sp, T* dst, const int dp,
    const T& border_value) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0); CHECK_GT(W, 0);
  // Check affine matrices
  CHECK_NOTNULL(M); CHECK_GT(Mn, 0);
  // Check source and dest images and pitches
  CHECK_NOTNULL(src); CHECK_GT(sp, 0);
  CHECK_NOTNULL(dst); CHECK_GT(dp, 0);

  std::vector<IppiWarpSpec*> pSpecs(N, nullptr);
  const IppiSize size{static_cast<int>(W), static_cast<int>(H)};
  const IppiPoint dstOffset = {0, 0};
  Ipp64f borderValue = border_value;

  #pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    // Copy affine matrix to array of arrays, format expected by IPP
    const int m = n % Mn;
    const double M_[2][3] = {
      { M[m * 6 + 0], M[m * 6 + 1], M[m * 6 + 2] },
      { M[m * 6 + 3], M[m * 6 + 4], M[m * 6 + 5] },
    };
    // Prepare spec for each image.
    int specSize = 0, initSize = 0;
    CHECK_IPP_CALL(ippiWarpAffineGetSize(
        size, size, internal::IppTraits<T>::data_type, M_, ippLinear, ippWarpForward,
        ippBorderConst, &specSize, &initSize));
    pSpecs[n] = CHECK_NOTNULL((IppiWarpSpec*)ippsMalloc_8u(specSize));
    CHECK_IPP_CALL(ippiWarpAffineLinearInit(
        size, size, internal::IppTraits<T>::data_type, M_, ippWarpForward, 1,
        ippBorderConst, &borderValue, 0, pSpecs[n]));
  }

  #pragma omp parallel for collapse(2)
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      int bufSize = 0;
      CHECK_IPP_CALL(ippiWarpGetBufferSize(pSpecs[n], size, &bufSize));
      Ipp8u* pBuffer = CHECK_NOTNULL((Ipp8u*)ippsMalloc_8u(bufSize));
      CHECK_IPP_CALL(internal::wrap_ippiWarpAffineLinear_C1R<T>(
          src + (n * C + c) * H * sp, sp * sizeof(T),
          dst + (n * C + c) * H * dp, dp * sizeof(T),
          dstOffset, size, pSpecs[n], pBuffer));
      ippsFree(pBuffer);
    }
  }

  #pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    ippsFree(pSpecs[n]);
  }
}

}  // namespace internal

template <typename T>
inline void affine_nchw_ipp(
    const int N, const int C, const int H, const int W,
    const int Mn, const double* M,
    const T* src, const int sp, T* dst, const int dp,
    const T& border_value) {
  // Note: IPP implementation can only be used for certain types.
  // By default, fall back to the generic implementation.
  affine_nchw_generic<T>(
      N, C, H, W, Mn, M, src, sp, dst, dp, border_value);
}

// Specialize affine_nchw_ipp<> to use the IPP implementation when supported.
#define DEFINE_IPP_SPECIALIZATION(T)                            \
  template <>                                                   \
  inline void affine_nchw_ipp<T>(                               \
      const int N, const int C, const int H, const int W,       \
      const int Mn, const double* M,                            \
      const T* src, const int sp, T* dst, const int dp,         \
      const T& border_value) {                                  \
    internal::affine_nchw_ipp<T>(                               \
        N, C, H, W, Mn, M, src, sp, dst, dp, border_value);     \
  }

DEFINE_IPP_SPECIALIZATION(double)
DEFINE_IPP_SPECIALIZATION(float)
DEFINE_IPP_SPECIALIZATION(int16_t)
DEFINE_IPP_SPECIALIZATION(uint8_t)
DEFINE_IPP_SPECIALIZATION(uint16_t)
#undef DEFINE_IPP_SPECIALIZATION

}  // namespace cpu
}  // namespace imgdistort

#endif  // IMGDISTORT_CPU_AFFINE_IPP_H_
