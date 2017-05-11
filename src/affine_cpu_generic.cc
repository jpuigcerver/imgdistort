#include <imgdistort/affine_cpu_generic.h>

#include <glog/logging.h>
#include <imgdistort/affine_util.h>
#include <imgdistort/interpolation.h>

namespace imgdistort {
namespace cpu {

template <typename T>
void affine_nchw_generic(
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
  // memory used to store the inverted affine matrices
  std::vector<double> _M(Mn * 6);
  // invert affine matrices
  #pragma omp parallel for
  for (int m = 0; m < Mn; ++m) {
    invert_affine_matrix(M + m * 6, _M.data() + m * 6);
  }
  // perform affine transformation
  #pragma omp parallel for collapse(4)
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          const int m = Mn > 1 ? n : 0;
          const double rx = _M[m*6+0] * x + _M[m*6+1] * y + _M[m*6+2];
          const double ry = _M[m*6+3] * x + _M[m*6+4] * y + _M[m*6+5];
          const int offsetSrc = (n * C + c) * H * sp;
          const int offsetDst = (n * C + c) * H * dp;
          dst[offsetDst + y * dp + x] =
              blinterp(src + offsetSrc, sp, H, W, ry, rx);
        }
      }
    }
  }
}

template
void affine_nchw_generic<int8_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int8_t* src, const int sp, int8_t* dst, const int dp);

template
void affine_nchw_generic<int16_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int16_t* src, const int sp, int16_t* dst, const int dp);

template
void affine_nchw_generic<int32_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int32_t* src, const int sp, int32_t* dst, const int dp);

template
void affine_nchw_generic<int64_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int64_t* src, const int sp, int64_t* dst, const int dp);

template
void affine_nchw_generic<uint8_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint8_t* src, const int sp, uint8_t* dst, const int dp);

template
void affine_nchw_generic<uint16_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint16_t* src, const int sp, uint16_t* dst, const int dp);

template
void affine_nchw_generic<uint32_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint32_t* src, const int sp, uint32_t* dst, const int dp);

template
void affine_nchw_generic<uint64_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint64_t* src, const int sp, uint64_t* dst, const int dp);

template
void affine_nchw_generic<float>(
    const int N, const int C, const int H, const int W, const double* M,
    const int Mn, const float* src, const int sp, float* dst, const int dp);

template
void affine_nchw_generic<double>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const double* src, const int sp, double* dst, const int dp);

}  // namespace cpu
}  // namespace imgdistort
