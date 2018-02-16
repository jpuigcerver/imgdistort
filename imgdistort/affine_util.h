#ifndef IMGDISTORT_AFFINE_UTIL_H_
#define IMGDISTORT_AFFINE_UTIL_H_

#include <imgdistort/config.h>

namespace imgdistort {

__host__ __device__
inline void invert_affine_matrix(const double M[6], double iM[6]) {
  const double M0 = M[0], M1 = M[1], M2 = M[2];
  const double M3 = M[3], M4 = M[4], M5 = M[5];
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

__host__ __device__
inline void invert_affine_matrix(const double M[2][3], double iM[2][3]) {
  const double M0 = M[0][0], M1 = M[0][1], M2 = M[0][2];
  const double M3 = M[1][0], M4 = M[1][1], M5 = M[1][2];
  const double D = (M0 * M4 != M1 * M3) ? 1.0 / (M0 * M4 - M1 * M3) : 0.0;
  const double A11 =  M4 * D;
  const double A12 = -M1 * D;
  const double A21 = -M3 * D;
  const double A22 =  M0 * D;
  const double b1 = -A11 * M2 - A12 * M5;
  const double b2 = -A21 * M2 - A22 * M5;
  iM[0][0] = A11; iM[0][1] = A12; iM[0][2] = b1;
  iM[1][0] = A21; iM[1][1] = A22; iM[1][2] = b2;
}

}  // namespace imgdistort

#endif  // IMGDISTORT_AFFINE_UTIL_H_
