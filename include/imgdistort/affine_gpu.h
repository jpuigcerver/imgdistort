#ifndef IMGDISTORT_AFFINE_GPU_H_
#define IMGDISTORT_AFFINE_GPU_H_

#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
namespace imgdistort {
namespace gpu {

template <typename T>
void affine_nchw(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const T* src, const int sp, T* dst, const int dp, cudaStream_t stream = 0);

}  // namespace gpu
}  // namespace imgdistort

extern "C" {
#endif  // __cplusplus

void imgdistort_gpu_affine_nchw_s8(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int8_t* src, const int sp, int8_t* dst, const int dp,
    cudaStream_t stream = 0);

void imgdistort_gpu_affine_nchw_s16(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int16_t* src, const int sp, int16_t* dst, const int dp,
    cudaStream_t stream = 0);

void imgdistort_gpu_affine_nchw_s32(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int32_t* src, const int sp, int32_t* dst, const int dp,
    cudaStream_t stream = 0);

void imgdistort_gpu_affine_nchw_s64(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int64_t* src, const int sp, int64_t* dst, const int dp,
    cudaStream_t stream = 0);

void imgdistort_gpu_affine_nchw_u8(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint8_t* src, const int sp, uint8_t* dst, const int dp,
    cudaStream_t stream = 0);

void imgdistort_gpu_affine_nchw_u16(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint16_t* src, const int sp, uint16_t* dst, const int dp,
    cudaStream_t stream = 0);

void imgdistort_gpu_affine_nchw_u32(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint32_t* src, const int sp, uint32_t* dst, const int dp,
    cudaStream_t stream = 0);

void imgdistort_gpu_affine_nchw_u64(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint64_t* src, const int sp, uint64_t* dst, const int dp,
    cudaStream_t stream = 0);

void imgdistort_gpu_affine_nchw_f32(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const float* src, const int sp, float* dst, const int dp,
    cudaStream_t stream = 0);

void imgdistort_gpu_affine_nchw_f64(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const double* src, const int sp, double* dst, const int dp,
    cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif

#endif  // IMGDISTORT_AFFINE_GPU_H_
