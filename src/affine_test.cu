#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "affine.cuh"
#include "base_test.cuh"

namespace testing {

MATCHER(FloatNearPointwise, "") {
  return Matcher<float>(FloatEq(get<1>(arg))).Matches(get<0>(arg));
}

}  // namespace testing

class AffineTest : public BaseTest {
 public:
  AffineTest() : BaseTest() {
    affine_cpu_ = new float [N * 6];
    CHECK_CUDA_CALL(cudaMalloc(&affine_gpu_, sizeof(float) * N * 6));
  }

  virtual void CopyToGPU() {
    BaseTest::CopyToGPU();
    CHECK_CUDA_CALL(cudaMemcpy(affine_gpu_, affine_cpu_, sizeof(float) * N * 6,
                               cudaMemcpyHostToDevice));
  }

  virtual ~AffineTest() {
    delete [] affine_cpu_;
    CHECK_CUDA_CALL(cudaFree(affine_gpu_));
  }

 protected:
  float *affine_cpu_, *affine_gpu_;
};

TEST_F(AffineTest, InvertAffineMatrix) {
  const std::vector<float> M{3.0, -2.0, 7.0, -3.0, -1.0, -2.0};
  std::vector<float> iM1(6, 0.0f), iM2(6, 0.0f);
  invert_affine_matrix(M.data(), iM1.data());
  invert_affine_matrix(iM1.data(), iM2.data());
  EXPECT_THAT(iM2, ::testing::Pointwise(::testing::FloatNearPointwise(), M));
}

TEST_F(AffineTest, AffineIdempotent) {
  affine_cpu_[0] = 1.0f; affine_cpu_[1]  = 0.0f; affine_cpu_[2]  = 0.0f;
  affine_cpu_[3] = 0.0f; affine_cpu_[4]  = 1.0f; affine_cpu_[5]  = 0.0f;
  affine_cpu_[6] = 1.0f; affine_cpu_[7]  = 0.0f; affine_cpu_[8]  = 0.0f;
  affine_cpu_[9] = 0.0f; affine_cpu_[10] = 1.0f; affine_cpu_[11] = 0.0f;
  CopyToGPU();
  affine_NCHW<float>(N, C, H, W, y_gpu(), x_gpu(), affine_gpu_);
  CopyToCPU();
  EXPECT_THAT(y_cpu_, ::testing::ElementsAreArray(x_cpu_));
}

TEST_F(AffineTest, AffineTranslate) {
  affine_cpu_[0] = 1.0f; affine_cpu_[1]  = 0.0f; affine_cpu_[2]  = +1.0f;
  affine_cpu_[3] = 0.0f; affine_cpu_[4]  = 1.0f; affine_cpu_[5]  = -2.0f;
  affine_cpu_[6] = 1.0f; affine_cpu_[7]  = 0.0f; affine_cpu_[8]  = -1.0f;
  affine_cpu_[9] = 0.0f; affine_cpu_[10] = 1.0f; affine_cpu_[11] = +2.0f;
  CopyToGPU();
  affine_NCHW<float>(N, C, H, W, y_gpu(), x_gpu(), affine_gpu_);
  CopyToCPU();
  const std::vector<float> expected_y{
    // Image 1
    0.0f, 5.0f,
    0.0f, 7.0f,
    0.0f, 0.0f,
    0.0f, 0.0f,

    0.0f, 13.0f,
    0.0f, 15.f,
    0.0f, 0.0f,
    0.0f, 0.0f,

    0.0f, 21.0f,
    0.0f, 23.0f,
    0.0f, 0.0f,
    0.0f, 0.0f,

    // Image 2
    0.0f, 0.0f,
    0.0f, 0.0f,
    26.0f, 0.0f,
    28.0f, 0.0f,

    0.0f, 0.0f,
    0.0f, 0.0f,
    34.0f, 0.0f,
    36.0f, 0.0f,

    0.0f, 0.0f,
    0.0f, 0.0f,
    42.0f, 0.0f,
    44.0f, 0.0f
  };
  EXPECT_THAT(y_cpu_, ::testing::Pointwise(::testing::FloatNearPointwise(),
                                           expected_y));
}

TEST_F(AffineTest, AffineScale) {
  memset(affine_cpu_, 0x00, sizeof(float) * 6 * 2);
  affine_cpu_[0] = 1.0f / 1.2f; affine_cpu_[1]  = 0.0f;
  affine_cpu_[3] = 0.0f;        affine_cpu_[4]  = 1.0f / 0.8f;
  affine_cpu_[6] = 1.0f / 0.7f; affine_cpu_[7]  = 0.0f;
  affine_cpu_[9] = 0.0f;        affine_cpu_[10] = 1.0f / 1.1f;
  CopyToGPU();
  affine_NCHW<float>(N, C, H, W, y_gpu(), x_gpu(), affine_gpu_);
  CopyToCPU();
  const std::vector<float> expected_y{
    // Image 1
    1.00f,  1.60f,
    2.60f,  2.88f,
    4.20f,  4.16f,
    5.80f,  5.44f,

    9.00f,  8.00f,
    10.60f, 9.28f,
    12.20f, 10.56f,
    13.80f, 11.84f,

    17.00f, 14.40f,
    18.60f, 15.68f,
    20.20f, 16.96f,
    21.80f, 18.24f,

    // Image 2
    25.00f, 25.70f,
    27.20f, 27.90f,
    29.40f, 30.10f,
    21.70f, 22.19f,

    33.00f, 33.70f,
    35.20f, 35.90f,
    37.40f, 38.10f,
    27.30f, 27.79f,

    41.00f, 41.70f,
    43.20f, 43.90f,
    45.40f, 46.10f,
    32.90f, 33.39f
  };
  EXPECT_THAT(y_cpu_, ::testing::Pointwise(::testing::FloatNearPointwise(),
                                           expected_y));
}

TEST_F(AffineTest, AffineRotate) {
}

TEST_F(AffineTest, AffineShear) {
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
