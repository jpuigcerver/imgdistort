#include <imgdistort/affine_cpu_ipp.h>

#include <vector>

#include <glog/logging.h>
#include <ipp.h>
#include <imgdistort/defines.h>

namespace imgdistort {

template <typename T>
inline IppStatus wrap_ippiWarpAffineLinear_C1R(
    const T* pSrc, int srcStep, T* pDst, int dstStep, IppiPoint dstRoiOffset,
    IppiSize dstRoiSize, IppiWarpSpec* pSpec, Ipp8u* pBuffer);

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

template <typename T, IppDataType dataType>
inline void affine_nchw_cpu_ipp(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const T* src, const int sp, T* dst, const int dp) {
  std::vector<IppiWarpSpec*> pSpecs(N, nullptr);
  const IppiSize size{W, H};
  Ipp64f borderValue = 0;

  for (int n = 0; n < N; ++n) {
    int specSize = 0, initSize = 0;
    CHECK_IPP_CALL(ippiWarpAffineGetSize(
        size, size, dataType, Mn > 1 ? M[n] : M[0], ippLinear, ippWarpForward,
        ippBorderConst, &specSize, &initSize));
    pSpecs[n] = CHECK_NOTNULL((IppiWarpSpec*)ippsMalloc_8u(specSize));
    CHECK_IPP_CALL(ippiWarpAffineLinearInit(
        size, size, dataType, Mn > 1 ? M[n] : M[0], ippWarpForward, 1,
        ippBorderConst, &borderValue, 0, pSpecs[n]));
  }

  #pragma omp parallel for collapse(2)
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      int bufSize = 0;
      CHECK_IPP_CALL(ippiWarpGetBufferSize(pSpecs[n], size, &bufSize));
      Ipp8u* pBuffer = CHECK_NOTNULL((Ipp8u*)ippsMalloc_8u(bufSize));
      const IppiPoint dstOffset = {0, 0};
      CHECK_IPP_CALL(wrap_ippiWarpAffineLinear_C1R<T>(
          src + (n * C + c) * H * sp, sp * sizeof(T),
          dst + (n * C + c) * H * dp, dp * sizeof(T),
          dstOffset, size, pSpecs[n], pBuffer));
      ippsFree(pBuffer);
    }
  }

  for (int n = 0; n < N; ++n) {
    ippsFree(pSpecs[n]);
  }
}

template <>
void affine_nchw_cpu_ipp<float>(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const float* src, const int sp, float* dst, const int dp) {
  affine_nchw_cpu_ipp<float, ipp32f>(N, C, H, W, M, Mn, src, sp, dst, dp);
}

template <>
void affine_nchw_cpu_ipp<double>(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const double* src, const int sp, double* dst, const int dp) {
  affine_nchw_cpu_ipp<double, ipp64f>(N, C, H, W, M, Mn, src, sp, dst, dp);
}

}  // namespace imgdistort
