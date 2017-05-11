#include <imgdistort/affine_gpu.h>

#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <imgdistort/affine_util.h>
#include <imgdistort/interpolation.h>
#include <imgdistort/defines.h>

namespace imgdistort {
namespace gpu {

template <typename T>
__global__
void kernel_affine_nchw(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn, const T* src, int sp, T* dst, int dp) {
  // All threads in the block work on the same image, and thus all use the same
  // affine matrix.
  // Copy affine transformation matrix into shared memory and invert it.
  // The affine matrix needs to be inverted because the kernel actually computes
  // the inverse operation.
  __shared__ double _M[6];

  for (int n = thGz; n < N; n += NTGz) {
    if (thBx == 0 && thBy == 0) {
      invert_affine_matrix(M + (n % Mn) * 6, _M);
    }
    __syncthreads();
    for (int c = thGy; c < C; c += NTGy) {
      const int offsetSrc = (n * C + c) * H * sp;
      const int offsetDst = (n * C + c) * H * dp;
      for (int i = thGx; i < H * W; i += NTGx) {
        const int x = i % W;
        const int y = i / W;
        const double rx = _M[0] * x + _M[1] * y + _M[2];
        const double ry = _M[3] * x + _M[4] * y + _M[5];
        dst[offsetDst + y * dp + x] =
            blinterp(src + offsetSrc, sp, H, W, ry, rx);
      }
    }
  }
}

// TODO(jpuigcerver): We are not using NVIDIA's NPPI warpAffine since it
// produces very different results from the generic implementation and
// Intel's IPPI. Investigate what are the reasons.
//
// TODO(jpuigcerver): Use textures to make the affine transform faster.
template <typename T>
void affine_nchw(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const T* src, const int sp, T* dst, const int dp, cudaStream_t stream) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0); CHECK_GT(W, 0);
  // Check affine matrices
  CHECK_NOTNULL(M); CHECK_GT(Mn, 0); CHECK(Mn == 1 || Mn == N);
  // Check source and dest images and pitches
  CHECK_NOTNULL(src); CHECK_GT(sp, 0);
  CHECK_NOTNULL(dst); CHECK_GT(dp, 0);
  // Launch kernel
  const dim3 block_size(1024, 1, 1);
  const dim3 grid_size(NUM_BLOCKS(H * W, 1024),
                       NUM_BLOCKS(C, 1),
                       NUM_BLOCKS(N, 1));
  kernel_affine_nchw<<<grid_size, block_size, 0, stream>>>(
      N, C, H, W, M, Mn, src, sp, dst, dp);
  if (stream == 0) { CHECK_LAST_CUDA_CALL(); }
}

}  // namespace gpu
}  // namespace imgdistort

#define DEFINE_C_FUNCTION(DESC, TYPE)                                   \
  extern "C" void imgdistort_gpu_affine_nchw_ ## DESC (                 \
      const int N, const int C, const int H, const int W,               \
      const double* M, const int Mn,                                    \
      const TYPE* src, const int sp, TYPE* dst, const int dp,           \
      cudaStream_t stream) {                                            \
    imgdistort::gpu::affine_nchw<TYPE>(                                 \
        N, C, H, W, M, Mn, src, sp, dst, dp, stream);                   \
  }

DEFINE_C_FUNCTION(s8,  int8_t)
DEFINE_C_FUNCTION(s16, int16_t)
DEFINE_C_FUNCTION(s32, int32_t)
DEFINE_C_FUNCTION(s64, int64_t)
DEFINE_C_FUNCTION(u8,  uint8_t)
DEFINE_C_FUNCTION(u16, uint16_t)
DEFINE_C_FUNCTION(u32, uint32_t)
DEFINE_C_FUNCTION(u64, uint64_t)
DEFINE_C_FUNCTION(f32, float)
DEFINE_C_FUNCTION(f64, double)
