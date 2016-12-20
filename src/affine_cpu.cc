#include <imgdistort/affine.h>
#include <glog/logging.h>

#ifdef WITH_IPP
#include <imgdistort/affine_cpu_ipp.h>
#else
#include <imgdistort/affine_cpu_generic.h>
#endif

namespace imgdistort {

template <typename T>
inline void affine_nchw_cpu(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const T* src, const int sp, T* dst, const int dp) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0); CHECK_GT(W, 0);
  // Check affine matrices
  CHECK_NOTNULL(M); CHECK_GT(Mn, 0); CHECK(Mn == 1 || Mn == N);
  // Check source and dest images and pitches
  CHECK_NOTNULL(src); CHECK_GT(sp, 0);
  CHECK_NOTNULL(dst); CHECK_GT(dp, 0);
#ifdef WITH_IPP
  affine_nchw_cpu_ipp<T>(N, C, H, W, M, Mn, src, sp, dst, dp);
#else
  affine_nchw_cpu_generic<T>(N, C, H, W, M, Mn, src, sp, dst, dp);
#endif
}

template <>
void affine_nchw<CPU, float>(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const float* src, const int sp, float* dst, const int dp) {
  affine_nchw_cpu<float>(N, C, H, W, M, Mn, src, sp, dst, dp);
}

template <>
void affine_nchw<CPU, double>(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const double* src, const int sp, double* dst, const int dp) {
  affine_nchw_cpu<double>(N, C, H, W, M, Mn, src, sp, dst, dp);
}

} // namespace imgdistort
