#ifndef IMGDISTORT_CPU_AFFINE_GENERIC_H_
#define IMGDISTORT_CPU_AFFINE_GENERIC_H_

#include <imgdistort/logging.h>
#include <imgdistort/affine_util.h>
#include <imgdistort/interpolation.h>

#include <vector>

namespace imgdistort {
namespace cpu {

template <typename T, typename Int>
void affine_nchw_generic(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const double* M,
    const T* src, const Int sp, T* dst, const Int dp,
    const T& border_value) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0); CHECK_GT(W, 0);
  // Check affine matrices
  CHECK_NOTNULL(M); CHECK_GT(Mn, 0);
  // Check source and dest images and pitches
  CHECK_NOTNULL(src); CHECK_GT(sp, 0);
  CHECK_NOTNULL(dst); CHECK_GT(dp, 0);
  // memory used to store the inverted affine matrices
  std::vector<double> _M(Mn * 6);
  // invert affine matrices
  #pragma omp parallel for
  for (Int m = 0; m < Mn; ++m) {
    invert_affine_matrix(M + m * 6, _M.data() + m * 6);
  }
  // perform affine transformation
  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (Int y = 0; y < H; ++y) {
        for (Int x = 0; x < W; ++x) {
          const Int m = n % Mn;
          const double rx = _M[m*6+0] * x + _M[m*6+1] * y + _M[m*6+2];
          const double ry = _M[m*6+3] * x + _M[m*6+4] * y + _M[m*6+5];
          const Int offsetSrc = (n * C + c) * H * sp;
          const Int offsetDst = (n * C + c) * H * dp;
          dst[offsetDst + y * dp + x] =
              blinterp(src + offsetSrc, sp, H, W, ry, rx, border_value);
        }
      }
    }
  }
}

}  // namespace cpu
}  // namespace imgdistort

#endif  // IMGDISTORT_CPU_AFFINE_GENERIC_H_
