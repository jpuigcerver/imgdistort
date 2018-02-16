#ifndef IMGDISTORT_CPU_AFFINE_BASE_TEST_H_
#define IMGDISTORT_CPU_AFFINE_BASE_TEST_H_

#include <imgdistort/testing/base.h>
#include <imgdistort/testing/image.h>

#include <vector>

#define AFFINE_TEST_IMG_W 32
#define AFFINE_TEST_IMG_H 16

namespace imgdistort {
namespace cpu {
namespace testing {

using imgdistort::testing::Image;

template<typename T>
std::vector<T> InitImage(const Image &);

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

}  // namespace testing
}  // namespace cpu
}  // namespace imgdistort

#endif // IMGDISTORT_CPU_AFFINE_BASE_TEST_H_
