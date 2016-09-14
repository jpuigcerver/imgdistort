#ifndef IMGDISTORT_SRC_AFFINE_H_
#define IMGDISTORT_SRC_AFFINE_H_

#include <cuda_runtime.h>

template <typename T>
__host__ __device__
inline void invert_affine_matrix(const T* M, T* iM) {
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

void affine_NCHW_f32(float* dst, const float* src,
                     const int N, const int C, const int H, const int W,
                     const float* M, const int Mn, cudaStream_t stream = 0);

void affine_NCHW_f64(double* dst, const double* src,
                     const int N, const int C, const int H, const int W,
                     const double* M, const int Mn, cudaStream_t stream = 0);

#endif  // IMGDISTORT_SRC_AFFINE_H_
