#include "morphology.h"

#include <glog/logging.h>

#include <limits>

#include "utils.h"

// W,H Block size
#define BLOCK_SIZE      16
// W,H Max Kernel sizes (up to 15 x 15 pixels)
#define MAX_KERNEL_SIZE 15
// W,H Max apron area size
#define MAX_APRON_SIZE  (BLOCK_SIZE + MAX_KERNEL_SIZE - 1)

template <typename T, bool dilate>
__global__
void kernel_dilate_or_erode_NCHW(
    T* D, const T* S, const int N, const int C, const int H, const int W,
    const bool* M, const int Mn, const int Mh, const int Mw, const T padv) {
  const int Bx = BLOCK_SIZE * (blockIdx.x % DIV_UP(W, BLOCK_SIZE));
  const int By = BLOCK_SIZE * blockIdx.y;
  const int n = blockIdx.z;
  const int c = blockIdx.x / DIV_UP(W, BLOCK_SIZE);
  const int x = Bx + threadIdx.x;
  const int y = By + threadIdx.y;
  const int offset_S = n * C * H * W + c * H * W;
  const int offset_M = (n % Mn) * Mh * Mw;

  // Copy structure kernel mask to shared memory.
  __shared__ bool _M[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
  if (threadIdx.x < Mw && threadIdx.y < Mh) {
    _M[threadIdx.x + threadIdx.y * Mw] =
        M[offset_M + threadIdx.y * Mw + threadIdx.x];
  }

  // Copy source image to shared memory
  __shared__ T _S[MAX_APRON_SIZE * MAX_APRON_SIZE];
  const int Aw = BLOCK_SIZE + Mw - 1, Ah = BLOCK_SIZE + Mh - 1;
  const int L = DIV_UP(Aw * Ah, BLOCK_SIZE * BLOCK_SIZE);
  for (int l = 0, j = L * thBi; l < L && j < Aw * Ah; ++l, ++j) {
    const int ax = j % Aw,           ay = j / Aw;
    const int sx = Bx + ax - Mw / 2, sy = By + ay - Mh / 2;
    _S[j] = pixv_pad(S + offset_S, sx, sy, W, H, padv);
  }
  __syncthreads();

  // Compute output pixel value
  if (x >= W || y >= H) return;
  T tmp = padv;
  for (int ki = 0; ki < Mh; ++ki) {
    for (int kj = 0; kj < Mw; ++kj) {
      if (!_M[ki * Mw + kj]) continue;
      const T v = pixv(_S, threadIdx.x + kj, threadIdx.y + ki, Aw, Ah);
      if ((dilate && tmp < v) || (!dilate && tmp > v)) tmp = v;
    }
  }
  D[offset_S + y * W + x] = tmp;
}

template <typename T, bool dilate>
void call_kernel_dilate_or_erode_NCHW(
    T* D, const T* S, const int N, const int C, const int H, const int W,
    const bool* M, const int Mn, const int Mh, const int Mw, const T padv,
    cudaStream_t stream) {
  CHECK_NOTNULL(D);
  CHECK_NOTNULL(S);
  CHECK_GT(N, 0);
  CHECK_GT(C, 0);
  CHECK_GT(H, 0);
  CHECK_GT(W, 0);
  CHECK_NOTNULL(M);
  CHECK_GT(Mn, 0);
  CHECK_GT(Mh, 0);
  CHECK_LE(Mh, MAX_KERNEL_SIZE);
  CHECK_GT(Mw, 0);
  CHECK_LE(Mw, MAX_KERNEL_SIZE);
  const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
  const dim3 grid_size(C * DIV_UP(W, BLOCK_SIZE), DIV_UP(H, BLOCK_SIZE), N);
  kernel_dilate_or_erode_NCHW<T, dilate><<<grid_size, block_size, 0, stream>>>(
      D, S, N, C, H, W, M, Mn, Mh, Mw, padv);
  CHECK_LAST_CUDA_CALL;
  if (stream == 0) {
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
  }
}

void dilate_NCHW_f32(
    float* dst, const float* src,
    const int N, const int C, const int H, const int W,
    const bool* M, const int Mn, const int Mh, const int Mw,
    cudaStream_t stream) {
  const float padv = -std::numeric_limits<float>::infinity();
  call_kernel_dilate_or_erode_NCHW<float, true>(dst, src, N, C, H, W,
                                                M, Mn, Mh, Mw, padv, stream);
}

void dilate_NCHW_f64(
    double* dst, const double* src,
    const int N, const int C, const int H, const int W,
    const bool* M, const int Mn, const int Mh, const int Mw,
    cudaStream_t stream) {
  const double padv = -std::numeric_limits<double>::infinity();
  call_kernel_dilate_or_erode_NCHW<double, true>(dst, src, N, C, H, W,
                                                 M, Mn, Mh, Mw, padv, stream);
}

void erode_NCHW_f32(
    float* dst, const float* src,
    const int N, const int C, const int H, const int W,
    const bool* M, const int Mn, const int Mh, const int Mw,
    cudaStream_t stream) {
  const float padv = std::numeric_limits<float>::infinity();
  call_kernel_dilate_or_erode_NCHW<float, false>(dst, src, N, C, H, W,
                                                 M, Mn, Mh, Mw, padv, stream);
}

void erode_NCHW_f64(
    double* dst, const double* src,
    const int N, const int C, const int H, const int W,
    const bool* M, const int Mn, const int Mh, const int Mw,
    cudaStream_t stream) {
  const double padv = std::numeric_limits<double>::infinity();
  call_kernel_dilate_or_erode_NCHW<double, false>(dst, src, N, C, H, W,
                                                  M, Mn, Mh, Mw, padv, stream);
}
