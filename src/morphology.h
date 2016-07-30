#ifndef IMGDISTORT_SRC_MORPHOLOGY_H_
#define IMGDISTORT_SRC_MORPHOLOGY_H_

#include <cuda_runtime.h>

void dilate_NCHW_f32(
    float* dst, const float* src,
    const int N, const int C, const int H, const int W,
    const bool* M, const int Mn, const int Mh, const int Mw,
    cudaStream_t stream = 0);

void dilate_NCHW_f64(
    double* dst, const double* src,
    const int N, const int C, const int H, const int W,
    const bool* M, const int Mn, const int Mh, const int Mw,
    cudaStream_t stream = 0);

void erode_NCHW_f32(
    float* dst, const float* src,
    const int N, const int C, const int H, const int W,
    const bool* M, const int Mn, const int Mh, const int Mw,
    cudaStream_t stream = 0);

void erode_NCHW_f64(
    double* dst, const double* src,
    const int N, const int C, const int H, const int W,
    const bool* M, const int Mn, const int Mh, const int Mw,
    cudaStream_t stream = 0);

#endif  // IMGDISTORT_SRC_MORPHOLOGY_H_
