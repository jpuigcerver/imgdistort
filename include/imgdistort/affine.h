#ifndef IMGDISTORT_AFFINE_H_
#define IMGDISTORT_AFFINE_H_

#include <imgdistort/defines.h>

EXTERNC void imgdistort_affine_nchw_cpu_float(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const float* src, const int sp, float* dst, const int dp);

EXTERNC void imgdistort_affine_nchw_cpu_double(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const double* src, const int sp, double* dst, const int dp);

EXTERNC void imgdistort_affine_nchw_gpu_float(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const float* src, const int sp, float* dst, const int dp);

EXTERNC void imgdistort_affine_nchw_gpu_double(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const double* src, const int sp, double* dst, const int dp);


#ifdef __cplusplus
namespace imgdistort {

template <DeviceType D, typename T>
void affine_nchw(
    const int N, const int C, const int H, const int W, const double M[][2][3],
    const int Mn, const T* src, const int sp, T* dst, const int dp);

}  // namespace imgdistort

#include <imgdistort/affine-inl.h>

#endif  // __cplusplus

#endif  // IMGDISTORT_AFFINE_H_
