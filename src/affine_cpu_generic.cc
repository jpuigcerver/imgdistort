#include <imgdistort/affine_cpu_generic.h>

#include <glog/logging.h>
#include <imgdistort/affine_util.h>
#include <imgdistort/interpolation.h>

namespace imgdistort {

template <typename T>
void affine_nchw_cpu_generic(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const T* src, const int sp, T* dst, const int dp) {
  // memory used to store the inverted affine matrices
  double _M[N][2][3];
  // invert affine matrices
  #pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    invert_affine_matrix(M[Mn > 1 ? n : 0], _M[n]);
  }
  // perform affine transformation
  #pragma omp parallel for collapse(4)
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          const double rx = _M[n][0][0] * x + _M[n][0][1] * y + _M[n][0][2];
          const double ry = _M[n][1][0] * x + _M[n][1][1] * y + _M[n][1][2];
          const int offsetSrc = n * C * H * sp + c * H * sp;
          const int offsetDst = n * C * H * dp + c * H * dp;
          dst[offsetDst + y * dp + x] =
              blinterp(src + offsetSrc, rx, ry, W, H, sp);
        }
      }
    }
  }
}

template
void affine_nchw_cpu_generic<float>(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const float* src, const int sp, float* dst, const int dp);

template
void affine_nchw_cpu_generic<double>(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const double* src, const int sp, double* dst, const int dp);

}  // namespace imgdistort
