#ifndef IMGDISTORT_GPU_AFFINE_BASE_TEST_H_
#define IMGDISTORT_GPU_AFFINE_BASE_TEST_H_

#include <imgdistort/cpu/affine_base_test.h>
#include <thrust/device_vector.h>

namespace imgdistort {
namespace gpu {
namespace testing {

template<typename T>
void OriginalTensor(const int N, const int C, thrust::device_vector<T> *output);

template<typename T>
const thrust::device_vector<T> &OriginalImage();

template<typename T>
const thrust::device_vector<T> &Affine1Image();

template<typename T>
const thrust::device_vector<T> &Affine2Image();

}  // namespace testing
}  // namespace gpu
}  // namespace imgdistort

#endif // IMGDISTORT_GPU_AFFINE_BASE_TEST_H_
