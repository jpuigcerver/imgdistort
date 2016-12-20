#include <imgdistort/affine.h>

#include <glog/logging.h>
#include <npp.h>

namespace imgdistort {

template <typename T>
inline NppStatus wrap_nppiWarpAffineLinear_C1R(
    const T* pSrc, const NppiSize& srcSize, const int srcStep,
    const NppiRect& srcROI, T* pDst, const int dstStep,
    const NppiRect& dstROI, const double coeffs[2][3]);

template <typename T>
inline NppStatus wrap_nppiWarpAffineLinear_P3R(
    const T* pSrc[3], const NppiSize& srcSize, const int srcStep,
    const NppiRect& srcROI, T* pDst[3], const int dstStep,
    const NppiRect& dstROI, const double coeffs[2][3]);

template <typename T>
inline NppStatus wrap_nppiWarpAffineLinear_P4R(
    const T* pSrc[4], const NppiSize& srcSize, const int srcStep,
    const NppiRect& srcROI, T* pDst[4], const int dstStep,
    const NppiRect& dstROI, const double coeffs[2][3]);

template <>
inline NppStatus wrap_nppiWarpAffineLinear_C1R<float>(
    const float* pSrc, const NppiSize& srcSize, const int srcStep,
    const NppiRect& srcROI, float* pDst, const int dstStep,
    const NppiRect& dstROI, const double coeffs[2][3]) {
  return nppiWarpAffine_32f_C1R(
      pSrc, srcSize, srcStep, srcROI, pDst, dstStep, dstROI,
      coeffs, NPPI_INTER_LINEAR);
}

template <>
inline NppStatus wrap_nppiWarpAffineLinear_C1R<double>(
    const double* pSrc, const NppiSize& srcSize, const int srcStep,
    const NppiRect& srcROI, double* pDst, const int dstStep,
    const NppiRect& dstROI, const double coeffs[2][3]) {
  return nppiWarpAffine_64f_C1R(
      pSrc, srcSize, srcStep, srcROI, pDst, dstStep, dstROI,
      coeffs, NPPI_INTER_LINEAR);
}

template <>
inline NppStatus wrap_nppiWarpAffineLinear_P3R<float>(
    const float* pSrc[3], const NppiSize& srcSize, const int srcStep,
    const NppiRect& srcROI, float* pDst[3], const int dstStep,
    const NppiRect& dstROI, const double coeffs[2][3]) {
  return nppiWarpAffine_32f_P3R(
      pSrc, srcSize, srcStep, srcROI, pDst, dstStep, dstROI,
      coeffs, NPPI_INTER_LINEAR);
}

template <>
inline NppStatus wrap_nppiWarpAffineLinear_P3R<double>(
    const double* pSrc[3], const NppiSize& srcSize, const int srcStep,
    const NppiRect& srcROI, double* pDst[3], const int dstStep,
    const NppiRect& dstROI, const double coeffs[2][3]) {
  return nppiWarpAffine_64f_P3R(
      pSrc, srcSize, srcStep, srcROI, pDst, dstStep, dstROI,
      coeffs, NPPI_INTER_LINEAR);
}

template <>
inline NppStatus wrap_nppiWarpAffineLinear_P4R<float>(
    const float* pSrc[4], const NppiSize& srcSize, const int srcStep,
    const NppiRect& srcROI, float* pDst[4], const int dstStep,
    const NppiRect& dstROI, const double coeffs[2][3]) {
  return nppiWarpAffine_32f_P4R(
      pSrc, srcSize, srcStep, srcROI, pDst, dstStep, dstROI,
      coeffs, NPPI_INTER_LINEAR);
}

template <>
inline NppStatus wrap_nppiWarpAffineLinear_P4R<double>(
    const double* pSrc[4], const NppiSize& srcSize, const int srcStep,
    const NppiRect& srcROI, double* pDst[4], const int dstStep,
    const NppiRect& dstROI, const double coeffs[2][3]) {
  return nppiWarpAffine_64f_P4R(
      pSrc, srcSize, srcStep, srcROI, pDst, dstStep, dstROI,
      coeffs, NPPI_INTER_LINEAR);
}

template <typename T>
void affine_nchw_gpu(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const T* src, const int sp, T* dst, const int dp) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0); CHECK_GT(W, 0);
  // Check affine matrices
  CHECK_NOTNULL(M); CHECK_GT(Mn, 0); CHECK(Mn == 1 || Mn == N);
  // Check source and dest images and pitches
  CHECK_NOTNULL(src); CHECK_GT(sp, 0);
  CHECK_NOTNULL(dst); CHECK_GT(dp, 0);

  const NppiSize size{W, H};
  const NppiRect roi{0, 0, W, H};
  if (C == 1) {
    for (int n = 0; n < N; ++n) {
      CHECK_NPP_CALL(wrap_nppiWarpAffineLinear_C1R(
          src + n * H * sp, size, sp * sizeof(T), roi,
          dst + n * H * dp, dp * sizeof(T), roi,
          M[Mn > 1 ? n : 0]));
    }
  } else if (C == 3) {
    for (int n = 0; n < N; ++n) {
      const T* srcPlanar[3] = {
        src + (n * C + 0) * H * sp,
        src + (n * C + 1) * H * sp,
        src + (n * C + 2) * H * sp
      };
      T* dstPlanar[3] = {
        dst + (n * C + 0) * H * dp,
        dst + (n * C + 1) * H * dp,
        dst + (n * C + 2) * H * dp
      };
      CHECK_NPP_CALL(wrap_nppiWarpAffineLinear_P3R(
          srcPlanar, size, sp * sizeof(T), roi,
          dstPlanar, dp * sizeof(T), roi,
          M[Mn > 1 ? n : 0]));
    }
  } else if (C == 4) {
    for (int n = 0; n < N; ++n) {
      const T* srcPlanar[4] = {
        src + (n * C + 0) * H * sp,
        src + (n * C + 1) * H * sp,
        src + (n * C + 2) * H * sp
      };
      T* dstPlanar[4] = {
        dst + (n * C + 0) * H * dp,
        dst + (n * C + 1) * H * dp,
        dst + (n * C + 2) * H * dp
      };
      CHECK_NPP_CALL(wrap_nppiWarpAffineLinear_P4R(
          srcPlanar, size, sp * sizeof(T), roi,
          dstPlanar, dp * sizeof(T), roi,
          M[Mn > 1 ? n : 0]));
    }
  } else {
  }
}


template <>
void affine_nchw<GPU, float>(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const float* src, const int sp, float* dst, const int dp) {
  affine_nchw_gpu<float>(N, C, H, W, M, Mn, src, sp, dst, dp);
}

template <>
void affine_nchw<GPU, double>(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const double* src, const int sp, double* dst, const int dp) {
  affine_nchw_gpu<double>(N, C, H, W, M, Mn, src, sp, dst, dp);
}

} // namespace imgdistort
