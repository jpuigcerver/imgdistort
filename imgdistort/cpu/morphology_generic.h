#ifndef IMGDISTORT_MORPHOLOGY_CPU_GENERIC_H_
#define IMGDISTORT_MORPHOLOGY_CPU_GENERIC_H_

#include <imgdistort/interpolation.h>
#include <imgdistort/logging.h>
#include <imgdistort/morphology_util.h>

#include <cstdint>

namespace imgdistort {
namespace cpu {
namespace internal {

template <typename T, typename Int, class Functor>
void morphology_nchw_generic(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const Int* Ms, const uint8_t* M,
    const T* src, const Int sp, T* dst, const Int dp) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0); CHECK_GT(W, 0);
  // Check transformation kernels
  CHECK_GT(Mn, 0); CHECK_NOTNULL(Ms); CHECK_NOTNULL(M);
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
  for (Int n = 1; n < Mn; ++n) {
    const Int Mh = Ms[2 * (n - 1) + 0];
    const Int Mw = Ms[2 * (n - 1) + 1];
    CHECK_GT(Mh, 0); CHECK_GT(Mw, 0);
    M_offset[n] = M_offset[n - 1] + Mh * Mw;
  }

  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (Int y = 0; y < H; ++y) {
        for (Int x = 0; x < W; ++x) {
          const Int Mh = Ms[2 * (n % Mn) + 0];
          const Int Mw = Ms[2 * (n % Mn) + 1];
          T tmp = pixv(src + (n * C + c) * H * sp, sp, y, x);
          for (Int ki = 0; ki < Mh; ++ki) {
            for (Int kj = 0; kj < Mw; ++kj) {
              const Int sy = y + ki - Mh / 2, sx = x + kj - Mw / 2;
              if (sy >= 0 && sx >= 0 && sy < H && sx < W &&
                  M[M_offset[n % Mn] + ki * Mw + kj] != 0) {
                tmp = Functor::f(tmp, pixv(src + (n * C + c) * H * sp, sp,
                                           y + ki - Mh / 2, x + kj - Mw / 2));
              }
            }
          }
          pixv(dst + (n * C + c) * H * dp, dp, y, x) = tmp;
        }
      }
    }
  }
}

}  // namespace internal

template <typename T, typename Int>
inline void dilate_nchw_generic(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const Int* Ms, const uint8_t* M,
    const T* src, const Int sp, T* dst, const Int dp) {
  using ::imgdistort::internal::DilateFunctor;
  internal::morphology_nchw_generic<T, Int, DilateFunctor<T>>(
      N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
}

template <typename T, typename Int>
inline void erode_nchw_generic(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const Int* Ms, const uint8_t* M,
    const T* src, const Int sp, T* dst, const Int dp) {
  using ::imgdistort::internal::ErodeFunctor;
  internal::morphology_nchw_generic<T, Int, ErodeFunctor<T>>(
      N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
}

}  // namespace cpu
}  // namespace imgdistort

#endif  // IMGDISTORT_MORPHOLOGY_CPU_GENERIC_H_
