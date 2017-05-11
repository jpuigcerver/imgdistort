#ifndef IMGDISTORT_MORPHOLOGY_CPU_H_
#define IMGDISTORT_MORPHOLOGY_CPU_H_

#include <cstdint>

#ifdef __cplusplus
namespace imgdistort {
namespace cpu {

template <typename T>
void dilate_nchw(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const T* src, const int sp, T* dst, const int dp);

template <typename T>
void erode_nchw(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const T* src, const int sp, T* dst, const int dp);

}  // namespace cpu
}  // namespace imgdistort

extern "C" {
#endif  // __cplusplus

void imgdistort_cpu_dilate_nchw_s8(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int8_t* src, const int sp, int8_t* dst, const int dp);

void imgdistort_cpu_dilate_nchw_s16(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int16_t* src, const int sp, int16_t* dst, const int dp);

void imgdistort_cpu_dilate_nchw_s32(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int32_t* src, const int sp, int32_t* dst, const int dp);

void imgdistort_cpu_dilate_nchw_s64(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int64_t* src, const int sp, int64_t* dst, const int dp);

void imgdistort_cpu_dilate_nchw_u8(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint8_t* src, const int sp, uint8_t* dst, const int dp);

void imgdistort_cpu_dilate_nchw_u16(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint16_t* src, const int sp, uint16_t* dst, const int dp);

void imgdistort_cpu_dilate_nchw_u32(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint32_t* src, const int sp, uint32_t* dst, const int dp);

void imgdistort_cpu_dilate_nchw_u64(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint64_t* src, const int sp, uint64_t* dst, const int dp);

void imgdistort_cpu_dilate_nchw_f32(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const float* src, const int sp, float* dst, const int dp);

void imgdistort_cpu_dilate_nchw_f64(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const double* src, const int sp, double* dst, const int dp);

void imgdistort_cpu_erode_nchw_s8(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int8_t* src, const int sp, int8_t* dst, const int dp);

void imgdistort_cpu_erode_nchw_s16(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int16_t* src, const int sp, int16_t* dst, const int dp);

void imgdistort_cpu_erode_nchw_s32(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int32_t* src, const int sp, int32_t* dst, const int dp);

void imgdistort_cpu_erode_nchw_s64(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const int64_t* src, const int sp, int64_t* dst, const int dp);

void imgdistort_cpu_erode_nchw_u8(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint8_t* src, const int sp, uint8_t* dst, const int dp);

void imgdistort_cpu_erode_nchw_u16(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint16_t* src, const int sp, uint16_t* dst, const int dp);

void imgdistort_cpu_erode_nchw_u32(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint32_t* src, const int sp, uint32_t* dst, const int dp);

void imgdistort_cpu_erode_nchw_u64(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const uint64_t* src, const int sp, uint64_t* dst, const int dp);

void imgdistort_cpu_erode_nchw_f32(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const float* src, const int sp, float* dst, const int dp);

void imgdistort_cpu_erode_nchw_f64(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const double* src, const int sp, double* dst, const int dp);

#ifdef __cplusplus
}
#endif

#endif  // IMGDISTORT_MORPHOLOGY_CPU_H_
