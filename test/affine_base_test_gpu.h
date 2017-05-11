#ifndef IMGDISTORT_AFFINE_GPU_BASE_TEST_H_
#define IMGDISTORT_AFFINE_GPU_BASE_TEST_H_

#include <thrust/device_vector.h>

namespace imgdistort {
namespace testing {
namespace gpu {

template<typename T>
void OriginalTensor(const int N, const int C, thrust::device_vector<T> *output);

template<typename T>
const thrust::device_vector<T> &OriginalImage();

template<typename T>
const thrust::device_vector<T> &Affine1Image();

template<typename T>
const thrust::device_vector<T> &Affine2Image();

}  // namespace gpu
}  // namespace testing
}  // namespace imgdistort

#endif // IMGDISTORT_AFFINE_GPU_BASE_TEST_H_
