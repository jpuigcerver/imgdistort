#include <imgdistort/morphology_cpu_generic.h>

#include <glog/logging.h>
#include <imgdistort/interpolation.h>

namespace imgdistort {
namespace cpu {

template <typename T>
struct MaxFunctor {
  static inline const T& f(const T& a, const T& b) {
    return std::max(a, b);
  }
};

template <typename T>
struct MinFunctor {
  static inline const T& f(const T& a, const T& b) {
    return std::min(a, b);
  }
};

template <typename T, class F>
static inline void morphology_nchw(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const T* src, const int sp, T* dst, const int dp, const T padv) {
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

  #pragma omp parallel for collapse(4)
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          const int Mh = Ms[2 * (Mn > 1 ? n : 0) + 0];
          const int Mw = Ms[2 * (Mn > 1 ? n : 0) + 1];
          T tmp = padv;
          for (int ki = 0; ki < Mh; ++ki) {
            for (int kj = 0; kj < Mw; ++kj) {
              if (M[M_offset[Mn > 1 ? n : 0] + ki * Mw + kj] != 0) {
                tmp = F::f(tmp, pixv(src + (n * C + c) * H * sp, sp, H, W,
                                     y + ki - Mh / 2, x + kj - Mw / 2, padv));
              }
            }
          }
          pixv(dst + (n * C + c) * H * dp, dp, y, x) = tmp;
        }
      }
    }
  }
}

template <typename T>
void dilate_nchw_generic(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const T* src, const int sp, T* dst, const int dp) {
  constexpr T padv = std::numeric_limits<T>::lowest();
  morphology_nchw<T, MaxFunctor<T>>(
      N, C, H, W, M, Ms, Mn, src, sp, dst, dp, padv);
}

template <typename T>
void erode_nchw_generic(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const T* src, const int sp, T* dst, const int dp) {
  const T padv = std::numeric_limits<T>::max();
  morphology_nchw<T, MinFunctor<T>>(
      N, C, H, W, M, Ms, Mn, src, sp, dst, dp, padv);
}

template
void dilate_nchw_generic<int8_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int8_t* src, const int sp, int8_t* dst, const int dp);

template
void dilate_nchw_generic<int16_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int16_t* src, const int sp, int16_t* dst, const int dp);

template
void dilate_nchw_generic<int32_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int32_t* src, const int sp, int32_t* dst, const int dp);

template
void dilate_nchw_generic<int64_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int64_t* src, const int sp, int64_t* dst, const int dp);

template
void dilate_nchw_generic<uint8_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint8_t* src, const int sp, uint8_t* dst, const int dp);

template
void dilate_nchw_generic<uint16_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint16_t* src, const int sp, uint16_t* dst, const int dp);

template
void dilate_nchw_generic<uint32_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint32_t* src, const int sp, uint32_t* dst, const int dp);

template
void dilate_nchw_generic<uint64_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint64_t* src, const int sp, uint64_t* dst, const int dp);

template
void dilate_nchw_generic<float>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const float* src, const int sp, float* dst, const int dp);

template
void dilate_nchw_generic<double>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const double* src, const int sp, double* dst, const int dp);

template
void erode_nchw_generic<int8_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int8_t* src, const int sp, int8_t* dst, const int dp);

template
void erode_nchw_generic<int16_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int16_t* src, const int sp, int16_t* dst, const int dp);

template
void erode_nchw_generic<int32_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int32_t* src, const int sp, int32_t* dst, const int dp);

template
void erode_nchw_generic<int64_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int64_t* src, const int sp, int64_t* dst, const int dp);

template
void erode_nchw_generic<uint8_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint8_t* src, const int sp, uint8_t* dst, const int dp);

template
void erode_nchw_generic<uint16_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint16_t* src, const int sp, uint16_t* dst, const int dp);

template
void erode_nchw_generic<uint32_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint32_t* src, const int sp, uint32_t* dst, const int dp);

template
void erode_nchw_generic<uint64_t>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint64_t* src, const int sp, uint64_t* dst, const int dp);

template
void erode_nchw_generic<float>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const float* src, const int sp, float* dst, const int dp);

template
void erode_nchw_generic<double>(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const double* src, const int sp, double* dst, const int dp);

}  // namespace cpu
}  // namespace imgdistort
