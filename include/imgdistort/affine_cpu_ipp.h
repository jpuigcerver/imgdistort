#ifndef IMGDISTORT_AFFINE_CPU_IPP_H_
#define IMGDISTORT_AFFINE_CPU_IPP_H_

#include <glog/logging.h>
#include <ippdefs.h>
#include <ippi.h>
#include <ipps.h>
#include <imgdistort/defines.h>

#include <cstdint>
#include <vector>

namespace imgdistort {
namespace cpu {

template <typename T>
inline IppStatus wrap_ippiWarpAffineLinear_C1R(
    const T* pSrc, int srcStep, T* pDst, int dstStep, IppiPoint dstRoiOffset,
    IppiSize dstRoiSize, IppiWarpSpec* pSpec, Ipp8u* pBuffer);

template <typename T, IppDataType dataType>
void affine_nchw_ipp(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const T* src, const int sp, T* dst, const int dp) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0); CHECK_GT(W, 0);
  // Check affine matrices
  CHECK_NOTNULL(M); CHECK_GT(Mn, 0); CHECK(Mn == 1 || Mn == N);
  // Check source and dest images and pitches
  CHECK_NOTNULL(src); CHECK_GT(sp, 0);
  CHECK_NOTNULL(dst); CHECK_GT(dp, 0);

  std::vector<IppiWarpSpec*> pSpecs(N, nullptr);
  const IppiSize size{W, H};
  const IppiPoint dstOffset = {0, 0};
  Ipp64f borderValue = 0;

  #pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    // Copy affine matrix to array of arrays, format expected by IPP
    const int m = Mn > 1 ? n : 0;
    const double M_[2][3] = {
      { M[m * 6 + 0], M[m * 6 + 1], M[m * 6 + 2] },
      { M[m * 6 + 3], M[m * 6 + 4], M[m * 6 + 5] },
    };
    // Prepare spec for each image.
    int specSize = 0, initSize = 0;
    CHECK_IPP_CALL(ippiWarpAffineGetSize(
        size, size, dataType, M_, ippLinear, ippWarpForward, ippBorderConst,
        &specSize, &initSize));
    pSpecs[n] = CHECK_NOTNULL((IppiWarpSpec*)ippsMalloc_8u(specSize));
    CHECK_IPP_CALL(ippiWarpAffineLinearInit(
        size, size, dataType, M_, ippWarpForward, 1, ippBorderConst,
        &borderValue, 0, pSpecs[n]));
  }

  #pragma omp parallel for collapse(2)
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      int bufSize = 0;
      CHECK_IPP_CALL(ippiWarpGetBufferSize(pSpecs[n], size, &bufSize));
      Ipp8u* pBuffer = CHECK_NOTNULL((Ipp8u*)ippsMalloc_8u(bufSize));
      CHECK_IPP_CALL(wrap_ippiWarpAffineLinear_C1R<T>(
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

template <typename T>
void affine_nchw_ipp(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const T* src, const int sp, T* dst, const int dp);

template <>
inline IppStatus wrap_ippiWarpAffineLinear_C1R<float>(
    const float* pSrc, int srcStep, float* pDst, int dstStep,
    IppiPoint dstRoiOffset, IppiSize dstRoiSize, IppiWarpSpec* pSpec,
    Ipp8u* pBuffer) {
  return ippiWarpAffineLinear_32f_C1R(
      pSrc, srcStep, pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiWarpAffineLinear_C1R<double>(
    const double* pSrc, int srcStep, double* pDst, int dstStep,
    IppiPoint dstRoiOffset, IppiSize dstRoiSize, IppiWarpSpec* pSpec,
    Ipp8u* pBuffer) {
  return ippiWarpAffineLinear_64f_C1R(
      pSrc, srcStep, pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiWarpAffineLinear_C1R<uint8_t>(
    const uint8_t* pSrc, int srcStep, uint8_t* pDst, int dstStep,
    IppiPoint dstRoiOffset, IppiSize dstRoiSize, IppiWarpSpec* pSpec,
    Ipp8u* pBuffer) {
  return ippiWarpAffineLinear_8u_C1R(
      pSrc, srcStep, pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiWarpAffineLinear_C1R<uint16_t>(
    const uint16_t* pSrc, int srcStep, uint16_t* pDst, int dstStep,
    IppiPoint dstRoiOffset, IppiSize dstRoiSize, IppiWarpSpec* pSpec,
    Ipp8u* pBuffer) {
  return ippiWarpAffineLinear_16u_C1R(
      pSrc, srcStep, pDst, dstStep, dstRoiOffset, dstRoiSize, pSpec, pBuffer);
}

template <>
void affine_nchw_ipp<uint8_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint8_t* src, const int sp, uint8_t* dst, const int dp) {
  affine_nchw_ipp<uint8_t, ipp8u>(N, C, H, W, M, Mn, src, sp, dst, dp);
}

template <>
void affine_nchw_ipp<uint16_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint16_t* src, const int sp, uint16_t* dst, const int dp) {
  affine_nchw_ipp<uint16_t, ipp16u>(N, C, H, W, M, Mn, src, sp, dst, dp);
}

template <>
void affine_nchw_ipp<float>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const float* src, const int sp, float* dst, const int dp) {
  affine_nchw_ipp<float, ipp32f>(N, C, H, W, M, Mn, src, sp, dst, dp);
}

template <>
void affine_nchw_ipp<double>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const double* src, const int sp, double* dst, const int dp) {
  affine_nchw_ipp<double, ipp64f>(N, C, H, W, M, Mn, src, sp, dst, dp);
}

}  // namespace cpu
}  // namespace imgdistort

#endif  // IMGDISTORT_AFFINE_CPU_IPP_H_
