#ifndef CUDA_KERNELS_AFFINE_CUH_
#define CUDA_KERNELS_AFFINE_CUH_
#include "utils.cuh"

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

  // Copy affine transformation matrix into shared memory. All thread in the
  // block work on the same image, and thus use the same affine transformation.
  __shared__ float M_i[2][3];
  if (threadIdx.x < 2 && threadIdx.y < 3) {
    M_i[threadIdx.x][threadIdx.y] = M[i * 6 + threadIdx.x * 3 + threadIdx.y];
  }
  __syncthreads();

  const float rx = M_i[0][0] * x + M_i[0][1] * y + M_i[0][2];
  const float ry = M_i[1][0] * x + M_i[1][1] * y + M_i[1][2];
  const int offset_ik = i * c * h * w + k * h * w;
  dst[offset_ik + y * w + x] = blinterp_border(src + offset_ik, rx, ry, w, h);
}

void invert_affine_matrix(const float* M, float* iM) {
  double D = M[0] * M[4] - M[1] * M[3];
  D = D != 0.0 ? 1.0/D : 0.0;
  const double A11 =  M[4] * D;
  const double A12 = -M[1] * D;
  const double A21 = -M[3] * D;
  const double A22 =  M[0] * D;
  const double b1 = -A11 * M[2] - A12 * M[5];
  const double b2 = -A21 * M[2] - A22 * M[5];
  iM[0] = A11; iM[1] = A12; iM[2] = b1;
  iM[3] = A21; iM[4] = A22; iM[5] = b2;
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
