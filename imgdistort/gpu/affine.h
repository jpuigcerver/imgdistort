#ifndef IMGDISTORT_GPU_AFFINE_H_
#define IMGDISTORT_GPU_AFFINE_H_

#include <cuda_runtime.h>
#include <imgdistort/affine_util.h>
#include <imgdistort/gpu/defines.h>
#include <imgdistort/interpolation.h>
#include <imgdistort/logging.h>

#include <cstdint>

#ifdef __cplusplus
namespace imgdistort {
namespace gpu {

namespace internal {
template <typename T, typename Int>
__global__
void kernel_affine_nchw(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const double* M, const T* src, Int sp, T* dst, Int dp,
    const T border_value) {
  // All threads in the block work on the same image, and thus all use the same
  // affine matrix.
  // Copy affine transformation matrix into shared memory and invert it.
  // The affine matrix needs to be inverted because the kernel actually computes
  // the inverse operation.
  __shared__ double _M[6];

  for (Int n = thGz; n < N; n += NTGz) {
    if (thBx == 0 && thBy == 0) {
      invert_affine_matrix(M + (n % Mn) * 6, _M);
    }
    __syncthreads();
    for (Int c = thGy; c < C; c += NTGy) {
      const Int offsetSrc = (n * C + c) * H * sp;
      const Int offsetDst = (n * C + c) * H * dp;
      for (Int i = thGx; i < H * W; i += NTGx) {
        const Int x = i % W;
        const Int y = i / W;
        const double rx = _M[0] * x + _M[1] * y + _M[2];
        const double ry = _M[3] * x + _M[4] * y + _M[5];
        dst[offsetDst + y * dp + x] =
            blinterp(src + offsetSrc, sp, H, W, ry, rx, border_value);
      }
    }
  }
}
}  // namespace internal

//
//
// @param N number of images in the batch
// @param C number of channels per image
// @param H height of the batch (maximum height among all images)
// @param W width of the batch (maximum width among all images)
// @param Mn number of affine matrices
// @param M matrices of the affine transformation (Mn x 2 x 3 array)
// @param src source batch
// @param sp source pitch (or stride)
// @param dst destination
// @param dp destination pitch (or stride)
template <typename T, typename Int>
inline void affine_nchw(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const double* M,
    const T* src, const Int sp, T* dst, const Int dp,
    const T& border_value = 0, cudaStream_t stream = 0) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0); CHECK_GT(W, 0);
  // Check affine matrices
  CHECK_NOTNULL(M); CHECK_GT(Mn, 0);
  // Check source and dest images and pitches
  CHECK_NOTNULL(src); CHECK_GT(sp, 0);
  CHECK_NOTNULL(dst); CHECK_GT(dp, 0);
  // Launch kernel
  const dim3 block_size(512, 1, 1);
  const dim3 grid_size(NUM_BLOCKS(H * W, 512),
                       NUM_BLOCKS(C, 1),
                       NUM_BLOCKS(N, 1));
  internal::kernel_affine_nchw<<<grid_size, block_size, 0, stream>>>(
      N, C, H, W, Mn, M, src, sp, dst, dp, border_value);
  if (stream == 0) { CHECK_LAST_CUDA_CALL(); }
}

}  // namespace gpu
}  // namespace imgdistort
#endif  // __cplusplus

// C bindings
#define DECLARE_BINDING(TYPE, SNAME)                              \
  EXTERN_C void imgdistort_gpu_affine_nchw_##SNAME(               \
      const int N, const int C, const int H, const int W,         \
      const int Mn, const double* M,                              \
      const TYPE* src, const int sp, TYPE* dst, const int dp,     \
      const TYPE border_value, cudaStream_t stream)

DECLARE_BINDING(float, f32);
DECLARE_BINDING(double, f64);
DECLARE_BINDING(int8_t, s8);
DECLARE_BINDING(int16_t, s16);
DECLARE_BINDING(int32_t, s32);
DECLARE_BINDING(int64_t, s64);
DECLARE_BINDING(uint8_t, u8);
DECLARE_BINDING(uint16_t, u16);
DECLARE_BINDING(uint32_t, u32);
DECLARE_BINDING(uint64_t, u64);
#undef DECLARE_BINDING

#endif  // IMGDISTORT_GPU_AFFINE_H_
