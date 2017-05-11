#include <imgdistort/morphology_cpu_ipp.h>

#include <vector>
#include <limits>

#include <glog/logging.h>
#include <ipp.h>
#include <imgdistort/defines.h>

namespace imgdistort {
namespace cpu {

template <typename T>
static inline IppStatus wrap_ippiMorphologyBorderGetSize(
    int roiWidth, IppiSize maskSize, int* pSpecSize, int* pBufferSize);

template <typename T>
static inline IppStatus wrap_ippiMorphologyBorderInit(
    int roiWidth, const Ipp8u* pMask, IppiSize maskSize,
    IppiMorphState* pMorphSpec, Ipp8u* pBuffer);

template <typename T>
static inline IppStatus wrap_ippiDilateBorder(
    const T* pSrc, int srcStep, T* pDst, int dstStep,
    IppiSize roiSize, IppiBorderType borderType, const T borderValue,
    const IppiMorphState* pMorphSpec, Ipp8u* pBuffer);

template <typename T>
static inline IppStatus wrap_ippiErodeBorder(
    const T* pSrc, int srcStep, T* pDst, int dstStep,
    IppiSize roiSize, IppiBorderType borderType, const T borderValue,
    const IppiMorphState* pMorphSpec, Ipp8u* pBuffer);


template <typename T, bool dilate>
void morphology_nchw_ipp(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const T* src, const int sp, T* dst, const int dp) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0);  CHECK_GT(W, 0);
  // Check transformation kernels
  CHECK_NOTNULL(M); CHECK_GT(Mn, 0); CHECK(Mn == 1 || Mn == N);
  CHECK_NOTNULL(Ms);
  // Check source and dest images and pitches
  CHECK_NOTNULL(src); CHECK_GT(sp, 0);
  CHECK_NOTNULL(dst); CHECK_GT(dp, 0);

  // Compute the offset of the kernel of each image.
  // offset_0 = 0
  // offset_1 = Mh_0 * Mw_0
  // offset_2 = Mh_0 * Mw_0 + Mh_1 * Mw_1
  // offset_i = offset_{i-1} + Mh_{i-1} * Mw_{i-1}
  // etc.
  std::vector<int> M_offset(Mn, 0);
  for (int n = 1; n < Mn; ++n) {
    const int Mh = Ms[2 * (n - 1) + 0];
    const int Mw = Ms[2 * (n - 1) + 1];
    CHECK_GT(Mh, 0); CHECK_GT(Mw, 0);
    M_offset[n] = M_offset[n - 1] + Mh * Mw;
  }

  #pragma omp parallel for collapse(2)
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      const int Mh = Ms[2 * (Mn > 1 ? n : 0) + 0];
      const int Mw = Ms[2 * (Mn > 1 ? n : 0) + 1];
      const IppiSize roiSize{W, H};
      const IppiSize maskSize{Mw, Mh};
      // Get spec and buffer sizes
      int specSize = 0, bufferSize = 0;
      CHECK_IPP_CALL(wrap_ippiMorphologyBorderGetSize<T>(
          W, maskSize, &specSize, &bufferSize));
      // Allocate memory for morphology spec and buffer
      IppiMorphState* pSpec = (IppiMorphState*)ippsMalloc_8u(specSize);
      Ipp8u* pBuffer = (Ipp8u*)ippsMalloc_8u(bufferSize);
      // Initialize morphology operation
      CHECK_IPP_CALL(wrap_ippiMorphologyBorderInit<T>(
          W, M + M_offset[Mn > 1 ? n : 0], maskSize, pSpec, pBuffer));
      if (dilate) {
        CHECK_IPP_CALL(wrap_ippiDilateBorder<T>(
            src + (n * C + c) * H * sp, sp * sizeof(T),
            dst + (n * C + c) * H * dp, dp * sizeof(T),
            roiSize, ippBorderConst, std::numeric_limits<T>::lowest(),
            pSpec, pBuffer));
      } else {
        CHECK_IPP_CALL(wrap_ippiErodeBorder<T>(
            src + (n * C + c) * H * sp, sp * sizeof(T),
            dst + (n * C + c) * H * dp, dp * sizeof(T),
            roiSize, ippBorderConst, std::numeric_limits<T>::max(),
            pSpec, pBuffer));
      }
    }
  }
}

template <>
inline IppStatus wrap_ippiMorphologyBorderGetSize<uint8_t>(
    int roiWidth, IppiSize maskSize, int* pSpecSize, int* pBufferSize) {
  return ippiMorphologyBorderGetSize_8u_C1R(
      roiWidth, maskSize, pSpecSize, pBufferSize);
}

template <>
inline IppStatus wrap_ippiMorphologyBorderGetSize<int16_t>(
    int roiWidth, IppiSize maskSize, int* pSpecSize, int* pBufferSize) {
  return ippiMorphologyBorderGetSize_16s_C1R(
      roiWidth, maskSize, pSpecSize, pBufferSize);
}

template <>
inline IppStatus wrap_ippiMorphologyBorderGetSize<uint16_t>(
    int roiWidth, IppiSize maskSize, int* pSpecSize, int* pBufferSize) {
  return ippiMorphologyBorderGetSize_16u_C1R(
      roiWidth, maskSize, pSpecSize, pBufferSize);
}

template <>
inline IppStatus wrap_ippiMorphologyBorderGetSize<float>(
    int roiWidth, IppiSize maskSize, int* pSpecSize, int* pBufferSize) {
  return ippiMorphologyBorderGetSize_32f_C1R(
      roiWidth, maskSize, pSpecSize, pBufferSize);
}

template <>
inline IppStatus wrap_ippiMorphologyBorderInit<uint8_t>(
    int roiWidth, const Ipp8u* pMask, IppiSize maskSize,
    IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiMorphologyBorderInit_8u_C1R(
      roiWidth, pMask, maskSize, pMorphSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiMorphologyBorderInit<int16_t>(
    int roiWidth, const Ipp8u* pMask, IppiSize maskSize,
    IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiMorphologyBorderInit_16s_C1R(
      roiWidth, pMask, maskSize, pMorphSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiMorphologyBorderInit<uint16_t>(
    int roiWidth, const Ipp8u* pMask, IppiSize maskSize,
    IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiMorphologyBorderInit_16u_C1R(
      roiWidth, pMask, maskSize, pMorphSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiMorphologyBorderInit<float>(
    int roiWidth, const Ipp8u* pMask, IppiSize maskSize,
    IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiMorphologyBorderInit_32f_C1R(
      roiWidth, pMask, maskSize, pMorphSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiDilateBorder<uint8_t>(
    const uint8_t* pSrc, int srcStep, uint8_t* pDst, int dstStep,
    IppiSize roiSize, IppiBorderType borderType, const uint8_t borderValue,
    const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiDilateBorder_8u_C1R(
      pSrc, srcStep, pDst, dstStep, roiSize, borderType, borderValue,
      pMorphSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiDilateBorder<int16_t>(
    const int16_t* pSrc, int srcStep, int16_t* pDst, int dstStep,
    IppiSize roiSize, IppiBorderType borderType, const int16_t borderValue,
    const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiDilateBorder_16s_C1R(
      pSrc, srcStep, pDst, dstStep, roiSize, borderType, borderValue,
      pMorphSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiDilateBorder<uint16_t>(
    const uint16_t* pSrc, int srcStep, uint16_t* pDst, int dstStep,
    IppiSize roiSize, IppiBorderType borderType, const uint16_t borderValue,
    const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiDilateBorder_16u_C1R(
      pSrc, srcStep, pDst, dstStep, roiSize, borderType, borderValue,
      pMorphSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiDilateBorder<float>(
    const float* pSrc, int srcStep, float* pDst, int dstStep,
    IppiSize roiSize, IppiBorderType borderType, const float borderValue,
    const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiDilateBorder_32f_C1R(
      pSrc, srcStep, pDst, dstStep, roiSize, borderType, borderValue,
      pMorphSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiErodeBorder<uint8_t>(
    const uint8_t* pSrc, int srcStep, uint8_t* pDst, int dstStep,
    IppiSize roiSize, IppiBorderType borderType, const uint8_t borderValue,
    const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiErodeBorder_8u_C1R(
      pSrc, srcStep, pDst, dstStep, roiSize, borderType, borderValue,
      pMorphSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiErodeBorder<int16_t>(
    const int16_t* pSrc, int srcStep, int16_t* pDst, int dstStep,
    IppiSize roiSize, IppiBorderType borderType, const int16_t borderValue,
    const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiErodeBorder_16s_C1R(
      pSrc, srcStep, pDst, dstStep, roiSize, borderType, borderValue,
      pMorphSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiErodeBorder<uint16_t>(
    const uint16_t* pSrc, int srcStep, uint16_t* pDst, int dstStep,
    IppiSize roiSize, IppiBorderType borderType, const uint16_t borderValue,
    const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiErodeBorder_16u_C1R(
      pSrc, srcStep, pDst, dstStep, roiSize, borderType, borderValue,
      pMorphSpec, pBuffer);
}

template <>
inline IppStatus wrap_ippiErodeBorder<float>(
    const float* pSrc, int srcStep, float* pDst, int dstStep,
    IppiSize roiSize, IppiBorderType borderType, const float borderValue,
    const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {
  return ippiErodeBorder_32f_C1R(
      pSrc, srcStep, pDst, dstStep, roiSize, borderType, borderValue,
      pMorphSpec, pBuffer);
}


#define DEFINE_IPP_IMPLEMENTATION(T)                                    \
  template <>                                                           \
  void dilate_nchw_ipp<T>(                                              \
      const int N, const int C, const int H, const int W,               \
      const uint8_t* M, const int* Ms, const int Mn,                    \
      const T* src, const int sp, T* dst, const int dp) {               \
    morphology_nchw_ipp<T, true>(                                       \
        N, C, H, W, M, Ms, Mn, src, sp, dst, dp);                       \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void erode_nchw_ipp<T>(                                               \
      const int N, const int C, const int H, const int W,               \
      const uint8_t* M, const int* Ms, const int Mn,                    \
      const T* src, const int sp, T* dst, const int dp) {               \
    morphology_nchw_ipp<T, false>(                                      \
      N, C, H, W, M, Ms, Mn, src, sp, dst, dp);                         \
  }


DEFINE_IPP_IMPLEMENTATION(int16_t)
DEFINE_IPP_IMPLEMENTATION(uint8_t)
DEFINE_IPP_IMPLEMENTATION(uint16_t)
DEFINE_IPP_IMPLEMENTATION(float)

}  // namespace cpu
}  // namespace imgdistort
