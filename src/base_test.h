#ifndef IMGDISTORT_SRC_BASE_TEST_H_
#define IMGDISTORT_SRC_BASE_TEST_H_

#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace testing {

MATCHER(FloatNearPointwise, "") {
  return Matcher<float>(FloatEq(get<1>(arg))).Matches(get<0>(arg));
}

}  // namespace testing


class BaseTest : public ::testing::Test {
 public:
  static const int N = 2, C = 3, H = 4, W = 2;

  BaseTest();
  virtual ~BaseTest();

  virtual void CopyToCPU();
  virtual void CopyToGPU();

  inline int x_size() const { return N * C * H * W * sizeof(float); }
  inline int y_size() const { return N * C * H * W * sizeof(float); }

 protected:
  std::vector<float> x_cpu_, y_cpu_;
  float *x_gpu_, *y_gpu_;
};

#endif  // IMGDISTORT_SRC_BASE_TEST_H_
