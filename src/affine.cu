#include "affine.h"

#include <glog/logging.h>

#include "utils.h"

#define BLOCK_SIZE 16

template <typename T>
__global__
void kernel_affine_NCHW(
    T* dst, const T* src, const int N, const int C, const int H, const int W,
    const T* M, const int Mn) {
  const int Bx = BLOCK_SIZE * (blockIdx.x % DIV_UP(W, BLOCK_SIZE));
  const int By = BLOCK_SIZE * blockIdx.y;
  const int n = blockIdx.z;
  const int c = blockIdx.x / DIV_UP(W, BLOCK_SIZE);
  const int x = Bx + threadIdx.x;
  const int y = By + threadIdx.y;

  // Copy affine transformation matrix into shared memory and invert it.
  // The affine matrix needs to be inverted because the kernel actually computes
  // the inverse operation.
  // All threads in the block work on the same image, and thus all use the same
  // affine matrix.
  __shared__ T _M[6];
  const int offset_M = (n % Mn) * 6;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    invert_affine_matrix(M + offset_M, _M);
  }
  __syncthreads();

  // Compute output pixel value
  if (x >= W || y >= H) return;
  const T rx = _M[0] * x + _M[1] * y + _M[2];
  const T ry = _M[3] * x + _M[4] * y + _M[5];
  const int offset_S = n * C * H * W + c * H * W;
  dst[offset_S + y * W + x] = blinterp(src + offset_S, rx, ry, W, H);
}

template <typename T>
void call_kernel_affine_NCHW(T* dst, const T* src,
                             const int N, const int C, const int H, const int W,
                             const T* M, const int Mn, cudaStream_t stream) {
  CHECK_NOTNULL(dst);
  CHECK_NOTNULL(src);
  CHECK_GT(N, 0);
  CHECK_GT(C, 0);
  CHECK_GT(H, 0);
  CHECK_GT(W, 0);
  CHECK_NOTNULL(M);
  CHECK_GT(Mn, 0);
  const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
  const dim3 grid_size(C * DIV_UP(W, BLOCK_SIZE), DIV_UP(H, BLOCK_SIZE), N);
  kernel_affine_NCHW<<<grid_size, block_size, 0, stream>>>(
      dst, src, N, C, H, W, M, Mn);
  CHECK_LAST_CUDA_CALL;
  if (stream == 0) {
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
  }
}

void affine_NCHW_f32(float* dst, const float* src,
                     const int N, const int C, const int H, const int W,
                     const float* M, const int Mn, cudaStream_t stream) {
  call_kernel_affine_NCHW<float>(dst, src, N, C, H, W, M, Mn, stream);
}

void affine_NCHW_f64(double* dst, const double* src,
                     const int N, const int C, const int H, const int W,
                     const double* M, const int Mn, cudaStream_t stream) {
  call_kernel_affine_NCHW<double>(dst, src, N, C, H, W, M, Mn, stream);
}
