#ifndef IMGDISTORT_AFFINE_CPU_IPP_H_
#define IMGDISTORT_AFFINE_CPU_IPP_H_

namespace imgdistort {
namespace cpu {

template <typename T>
void affine_nchw_ipp(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const T* src, const int sp, T* dst, const int dp);

}  // namespace cpu
}  // namespace imgdistort

#endif  // IMGDISTORT_AFFINE_CPU_IPP_H_