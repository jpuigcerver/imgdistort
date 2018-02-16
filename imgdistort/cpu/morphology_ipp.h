#ifndef IMGDISTORT_MORPHOLOGY_CPU_IPP_H_
#define IMGDISTORT_MORPHOLOGY_CPU_IPP_H_

#include <imgdistort/cpu/morphology_ipp_operation.h>
#include <imgdistort/logging.h>
#include <ipp.h>

#include <cstdint>
#include <vector>

namespace imgdistort {
namespace cpu {
namespace internal {

template <typename T, typename Operation>
void morphology_nchw_ipp(
    const int N, const int C, const int H, const int W,
    const int Mn, const int* Ms, const uint8_t* M,
    const T* src, const int sp, T* dst, const int dp) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0); CHECK_GT(W, 0);
  // Check transformation kernels
  CHECK_NOTNULL(M); CHECK_GT(Mn, 0);
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
    const int Mh = Ms[2 * (n - 1)    ];
    const int Mw = Ms[2 * (n - 1) + 1];
    CHECK_GT(Mh, 0); CHECK_GT(Mw, 0);
    M_offset[n] = M_offset[n - 1] + Mh * Mw;
  }

  #pragma omp parallel for collapse(2)
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      const int Mh = Ms[2 * (n % Mn)    ];
      const int Mw = Ms[2 * (n % Mn) + 1];
      const IppiSize roiSize{W, H};
      Operation op;
      op.Initialize(W, Mw, Mh, M + M_offset[n % Mn]);
      CHECK_IPP_CALL(op(src + (n * C + c) * H * sp, sp * sizeof(T),
                        dst + (n * C + c) * H * dp, dp * sizeof(T),
                        roiSize, op.Spec(), op.Buffer()));
    }
  }
}

template <typename T>
inline void dilate_nchw_ipp(
    const int N, const int C, const int H, const int W,
    const int Mn, const int* Ms, const uint8_t* M,
    const T* src, const int sp, T* dst, const int dp) {
  morphology_nchw_ipp<T, DilateOperation<T>>(
      N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
}

template <typename T>
inline void erode_nchw_ipp(
    const int N, const int C, const int H, const int W,
    const int Mn, const int* Ms, const uint8_t* M,
    const T* src, const int sp, T* dst, const int dp) {
  morphology_nchw_ipp<T, ErodeOperation<T>>(
      N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
}

}  // namespace internal

template <typename T>
inline void dilate_nchw_ipp(
    const int N, const int C, const int H, const int W,
    const int Mn, const int* Ms, const uint8_t* M,
    const T* src, const int sp, T* dst, const int dp) {
  // Note: IPP implementation can only be used for certain types.
  // By default, fall back to the generic implementation.
  dilate_nchw_generic<T, int>(N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
}

template <typename T>
inline void erode_nchw_ipp(
    const int N, const int C, const int H, const int W,
    const int Mn, const int* Ms, const uint8_t* M,
    const T* src, const int sp, T* dst, const int dp) {
  // Note: IPP implementation can only be used for certain types.
  // By default, fall back to the generic implementation.
  erode_nchw_generic<T, int>(N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
}

// Specialize affine_nchw_ipp<> to use the IPP implementation when supported.
#define DEFINE_IPP_SPECIALIZATION(T)                                    \
  template <>                                                           \
  inline void dilate_nchw_ipp<T>(                                       \
      const int N, const int C, const int H, const int W,               \
      const int Mn, const int* Ms, const uint8_t* M,                    \
      const T* src, const int sp, T* dst, const int dp) {               \
    internal::dilate_nchw_ipp<T>(N, C, H, W, Mn, Ms, M, src, sp, dst, dp); \
  }                                                                     \
                                                                        \
  template <>                                                           \
  inline void erode_nchw_ipp<T>(                                        \
      const int N, const int C, const int H, const int W,               \
      const int Mn, const int* Ms, const uint8_t* M,                    \
      const T* src, const int sp, T* dst, const int dp) {               \
    internal::erode_nchw_ipp<T>(N, C, H, W, Mn, Ms, M, src, sp, dst, dp); \
  }

DEFINE_IPP_SPECIALIZATION(float)
DEFINE_IPP_SPECIALIZATION(int16_t)
DEFINE_IPP_SPECIALIZATION(uint8_t)
DEFINE_IPP_SPECIALIZATION(uint16_t)
#undef DEFINE_IPP_SPECIALIZATION

}  // namespace cpu
}  // namespace imgdistort

#endif  // IMGDISTORT_MORPHOLOGY_CPU_IPP_H_
