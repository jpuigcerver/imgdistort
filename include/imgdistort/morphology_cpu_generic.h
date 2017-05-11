#ifndef IMGDISTORT_MORPHOLOGY_CPU_GENERIC_H_
#define IMGDISTORT_MORPHOLOGY_CPU_GENERIC_H_

#include <cstdint>

namespace imgdistort {
namespace cpu {

template <typename T>
void dilate_nchw_generic(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const T* src, const int sp, T* dst, const int dp);

template <typename T>
void erode_nchw_generic(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const T* src, const int sp, T* dst, const int dp);

}  // namespace cpu
}  // namespace imgdistort

#endif  // IMGDISTORT_MORPHOLOGY_CPU_GENERIC_H_
