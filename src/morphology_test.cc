#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "base_test.h"
#include "morphology.h"
#include "utils.h"

class MorphologyTest : public BaseTest {
 public:
  const int KW = 5, KH = 3;

  MorphologyTest() : BaseTest() {
    M_cpu_ = new bool [N * KW * KH];
    memset(M_cpu_, 0x00, N * KW * KH);
    CHECK_CUDA_CALL(cudaMalloc(&M_gpu_, N * KW * KH));
  }

  virtual ~MorphologyTest() {
    delete [] M_cpu_;
    CHECK_CUDA_CALL(cudaFree(M_gpu_));
  }

  virtual void CopyToGPU() {
    BaseTest::CopyToGPU();
    CHECK_CUDA_CALL(cudaMemcpy(M_gpu_, M_cpu_, N * KW * KH,
                               cudaMemcpyHostToDevice));
  }

 protected:
  bool *M_cpu_, *M_gpu_;
};

TEST_F(MorphologyTest, Idempotent) {
  M_cpu_[          (KH / 2) * KW + (KW / 2)] = true;
  M_cpu_[KW * KH + (KH / 2) * KW + (KW / 2)] = true;
  CopyToGPU();
  // First try dilate
  dilate_NCHW_f32(y_gpu_, x_gpu_, N, C, H, W, M_gpu_, N, KH, KW);
  CopyToCPU();
  EXPECT_THAT(y_cpu_, ::testing::ElementsAreArray(x_cpu_));
  // Then try erode
  erode_NCHW_f32(y_gpu_, x_gpu_, N, C, H, W, M_gpu_, N, KH, KW);
  CopyToCPU();
  EXPECT_THAT(y_cpu_, ::testing::ElementsAreArray(x_cpu_));
}

TEST_F(MorphologyTest, Dilate) {
  M_cpu_[            (KH / 2 - 1) * KW + (KW / 2)] = true;
  M_cpu_[            (KH / 2    ) * KW + (KW / 2)] = true;
  M_cpu_[            (KH / 2 + 1) * KW + (KW / 2)] = true;
  M_cpu_[(KW * KH) + (KH / 2) * KW + (KW / 2 - 1)] = true;
  M_cpu_[(KW * KH) + (KH / 2) * KW + (KW / 2    )] = true;
  M_cpu_[(KW * KH) + (KH / 2) * KW + (KW / 2 + 1)] = true;
  CopyToGPU();
  dilate_NCHW_f32(y_gpu_, x_gpu_, N, C, H, W, M_gpu_, N, KH, KW);
  CopyToCPU();
    const std::vector<float> expected_y{
    // Image 1
    3.0f,  4.0f,
    5.0f,  6.0f,
    7.0f,  8.0f,
    7.0f,  8.0f,

    11.0f, 12.0f,
    13.0f, 14.0f,
    15.0f, 16.0f,
    15.0f, 16.0f,

    19.0f, 20.0f,
    21.0f, 22.0f,
    23.0f, 24.0f,
    23.0f, 24.0f,

    // Image 2
    26.0f, 26.0f,
    28.0f, 28.0f,
    30.0f, 30.0f,
    32.0f, 32.0f,

    34.0f, 34.0f,
    36.0f, 36.0f,
    38.0f, 38.0f,
    40.0f, 40.0f,

    42.0f, 42.0f,
    44.0f, 44.0f,
    46.0f, 46.0f,
    48.0f, 48.0f
  };
  EXPECT_THAT(y_cpu_, ::testing::Pointwise(::testing::FloatNearPointwise(),
                                           expected_y));
}

TEST_F(MorphologyTest, Erode) {
  M_cpu_[            (KH / 2 - 1) * KW + (KW / 2)] = true;
  M_cpu_[            (KH / 2    ) * KW + (KW / 2)] = true;
  M_cpu_[            (KH / 2 + 1) * KW + (KW / 2)] = true;
  M_cpu_[(KW * KH) + (KH / 2) * KW + (KW / 2 - 1)] = true;
  M_cpu_[(KW * KH) + (KH / 2) * KW + (KW / 2    )] = true;
  M_cpu_[(KW * KH) + (KH / 2) * KW + (KW / 2 + 1)] = true;
  CopyToGPU();
  erode_NCHW_f32(y_gpu_, x_gpu_, N, C, H, W, M_gpu_, N, KH, KW);
  CopyToCPU();
  const std::vector<float> expected_y{
    // Image 1
    1.0f,  2.0f,
    1.0f,  2.0f,
    3.0f,  4.0f,
    5.0f,  6.0f,

    9.0f,  10.0f,
    9.0f,  10.0f,
    11.0f, 12.0f,
    13.0f, 14.0f,

    17.0f, 18.0f,
    17.0f, 18.0f,
    19.0f, 20.0f,
    21.0f, 22.0f,

    // Image 2
    25.0f, 25.0f,
    27.0f, 27.0f,
    29.0f, 29.0f,
    31.0f, 31.0f,

    33.0f, 33.0f,
    35.0f, 35.0f,
    37.0f, 37.0f,
    39.0f, 39.0f,

    41.0f, 41.0f,
    43.0f, 43.0f,
    45.0f, 45.0f,
    47.0f, 47.0f
  };
  EXPECT_THAT(y_cpu_, ::testing::Pointwise(::testing::FloatNearPointwise(),
                                           expected_y));
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
