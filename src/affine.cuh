#ifndef CUDA_KERNELS_AFFINE_CUH_
#define CUDA_KERNELS_AFFINE_CUH_
#include "utils.cuh"

__host__ __device__
void invert_affine_matrix(const float* M, float* iM) {
  const double M0 = M[0], M1 = M[1], M2 = M[2], M3 = M[3], M4 = M[4], M5 = M[5];
  const double D = (M0 * M4 != M1 * M3) ? 1.0 / (M0 * M4 - M1 * M3) : 0.0;
  const double A11 =  M4 * D;
  const double A12 = -M1 * D;
  const double A21 = -M3 * D;
  const double A22 =  M0 * D;
  const double b1 = -A11 * M2 - A12 * M5;
  const double b2 = -A21 * M2 - A22 * M5;
  iM[0] = A11; iM[1] = A12; iM[2] = b1;
  iM[3] = A21; iM[4] = A22; iM[5] = b2;
}

template <typename T>
__global__
void kernel_affine_NCHW(const int n, const int c, const int h, const int w,
                        T* dst, const T* src, const float* M) {
  const int NTcw = blockDim.x * DIV_UP(c * w, blockDim.x);
  const int i = thGx / NTcw;
  const int k = (thGx % NTcw) / w;
  const int x = (thGx % NTcw) % w;
  const int y = thGy;
  if (y >= h || x >= w || k >= c || i >= n) return;

  // Copy affine transformation matrix into shared memory and invert it.
  // The affine matrix needs to be inverted because the kernel actually computes
  // the inverse operation.
  // All threads in the block work on the same image, and thus all use the same
  // affine matrix.
  __shared__ float M_i[6];
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    invert_affine_matrix(M + i * 6, M_i);
  }
  __syncthreads();

  const float rx = M_i[0] * x + M_i[1] * y + M_i[2];
  const float ry = M_i[3] * x + M_i[4] * y + M_i[5];
  const int offset_ik = i * c * h * w + k * h * w;
  dst[offset_ik + y * w + x] = blinterp(src + offset_ik, rx, ry, w, h);
}

template <typename T>
void affine_NCHW(const int n, const int c, const int h, const int w,
                 T* dst, const T* src, const float* M,
                 cudaStream_t stream = 0) {
  CHECK_NOTNULL(dst);
  CHECK_NOTNULL(src);
  CHECK_NOTNULL(M);
  const dim3 block_size(16, 16);
  const dim3 grid_size(n * DIV_UP(c * w, 16), DIV_UP(h, 16));
  kernel_affine_NCHW<<<grid_size, block_size, 0, stream>>>(
      n, c, h, w, dst, src, M);
  CHECK_LAST_CUDA_CALL;
  if (stream == 0) {
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
  }
}

#endif  // CUDA_KERNELS_AFFINE_CUH_
