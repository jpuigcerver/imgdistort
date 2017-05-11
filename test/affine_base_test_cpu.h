#ifndef IMGDISTORT_AFFINE_CPU_BASE_TEST_H_
#define IMGDISTORT_AFFINE_CPU_BASE_TEST_H_

#include <vector>

#include "test_image.h"

namespace imgdistort {
namespace testing {
namespace cpu {

template<typename T>
std::vector<T> InitImage(const TestImage &);

template<typename T>
void OriginalTensor(const int N, const int C, std::vector<T> *original);

template<typename T>
void ExpectedGenericTensor(
    const int N, const int C, std::vector<T> *expected);

template<typename T>
const std::vector<T> &OriginalImage();

template<typename T>
const std::vector<T> &Affine1Image();

template<typename T>
const std::vector<T> &Affine2Image();

}  // namespace cpu
}  // namespace testing
}  // namespace imgdistort

#endif // IMGDISTORT_AFFINE_CPU_BASE_TEST_H_
