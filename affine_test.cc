#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "affine.h"
#include "base_test.h"
#include "utils.h"

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
  const std::vector<float> M{3.0f, -2.0f, 7.0f, -3.0f, -1.0f, -2.0f};
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
  affine_NCHW_f32(y_gpu_, x_gpu_, N, C, H, W, affine_gpu_, N);
  CopyToCPU();
  EXPECT_THAT(y_cpu_, ::testing::ElementsAreArray(x_cpu_));
}

TEST_F(AffineTest, AffineTranslate) {
  affine_cpu_[0] = 1.0f; affine_cpu_[1]  = 0.0f; affine_cpu_[2]  = +1.0f;
  affine_cpu_[3] = 0.0f; affine_cpu_[4]  = 1.0f; affine_cpu_[5]  = -2.0f;
  affine_cpu_[6] = 1.0f; affine_cpu_[7]  = 0.0f; affine_cpu_[8]  = -1.0f;
  affine_cpu_[9] = 0.0f; affine_cpu_[10] = 1.0f; affine_cpu_[11] = +2.0f;
  CopyToGPU();
  affine_NCHW_f32(y_gpu_, x_gpu_, N, C, H, W, affine_gpu_, N);
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
  affine_cpu_[0] = 1.25f; affine_cpu_[1]  = 0.0f;
  affine_cpu_[3] = 0.0f;  affine_cpu_[4]  = 0.8f;
  affine_cpu_[6] = 0.7f;  affine_cpu_[7]  = 0.0f;
  affine_cpu_[9] = 0.0f;  affine_cpu_[10] = 1.4f;
  CopyToGPU();
  affine_NCHW_f32(y_gpu_, x_gpu_, N, C, H, W, affine_gpu_, N);
  CopyToCPU();
  const std::vector<float> expected_y{
    // Image 1
    1.00f,   1.80f,
    3.50f,   4.30f,
    6.00f,   6.80f,
    7.00f,   7.80f,

    9.00f,   9.80f,
    11.50f, 12.30f,
    14.00f, 14.80f,
    15.00f, 15.80f,

    17.00f, 17.80f,
    19.50f, 20.30f,
    22.00f, 22.80f,
    23.00f, 23.80f,

    // Image 2
    25.0000f, 26.0000f,
    26.4286f, 27.4286f,
    27.8571f, 28.8571f,
    29.2857f, 30.2857f,

    26.4286f, 26.4286f,
    35.20f, 35.90f,
    37.40f, 38.10f,
    27.30f, 27.79f,

    41.00f, 41.70f,
    43.20f, 43.90f,
    45.40f, 46.10f,
    32.90f, 33.39f
  };
  // TODO(jpuigcerver): Fix this, the test is old.
  /*EXPECT_THAT(y_cpu_, ::testing::Pointwise(::testing::FloatNearPointwise(),
    expected_y));*/
}

TEST_F(AffineTest, AffineRotate) {
  // TODO(jpuigcerver): Implement
}

TEST_F(AffineTest, AffineShear) {
  // TODO(jpuigcerver): Implement
}
