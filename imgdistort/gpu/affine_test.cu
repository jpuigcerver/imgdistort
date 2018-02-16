#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>  // IWYU pragma: keep
#include <thrust/device_vector.h>

#include <imgdistort/gpu/affine.h>
#include <imgdistort/gpu/affine_base_test.h>
#include <imgdistort/testing/base.h>

#include <cstdint>
#include <vector>

typedef std::tuple<int, int> TestArgs;

#define REGISTER_TYPED_TESTS(DESC, TYPE)                                \
  class AffineGPUTest_##DESC :                                          \
      public ::testing::TestWithParam<TestArgs> {};                     \
                                                                        \
  TEST_P(AffineGPUTest_##DESC, Idempotent) {                            \
    const TestArgs param = GetParam();                                  \
    const int N = std::get<0>(param);                                   \
    const int C = std::get<1>(param);                                   \
    const thrust::device_vector<double> M(                              \
        std::vector<double>{                                            \
          1.0, 0.0, 0.0,                                                \
          0.0, 1.0, 0.0                                                 \
        });                                                             \
    thrust::device_vector<TYPE> input;                                  \
    imgdistort::gpu::testing::OriginalTensor<TYPE>(N, C, &input);       \
    /* Check C++ API */                                                 \
    thrust::device_vector<TYPE> output(input.size());                   \
    imgdistort::gpu::affine_nchw<TYPE, int>(                            \
        N, C, 16, 32, 1, M.data().get(), input.data().get(), 32,        \
        output.data().get(), 32, 0);                                    \
    /* Check C API */                                                   \
    thrust::device_vector<TYPE> output2(input.size());                  \
    imgdistort_gpu_affine_nchw_##DESC(                                  \
        N, C, 16, 32, 1, M.data().get(), input.data().get(), 32,        \
        output2.data().get(), 32, 0, nullptr);                          \
    thrust::host_vector<TYPE> input_cpu(input);                         \
    thrust::host_vector<TYPE> output_cpu(output);                       \
    thrust::host_vector<TYPE> output2_cpu(output2);                     \
    EXPECT_THAT(output_cpu,  ::testing::ElementsAreArray(input_cpu));   \
    EXPECT_THAT(output2_cpu, ::testing::ElementsAreArray(input_cpu));   \
  }                                                                     \
                                                                        \
  TEST_P(AffineGPUTest_##DESC, Generic) {                               \
    const TestArgs param = GetParam();                                  \
    const int N = std::get<0>(param);                                   \
    const int C = std::get<1>(param);                                   \
    const thrust::device_vector<double> M(                              \
        std::vector<double>{                                            \
          /* Matrix 1 */                                                \
          +0.76, +0.83, -0.18,                                          \
          -0.05, +0.78, +1.18,                                          \
          /* Matrix 2 */                                                \
          -1.40, +0.70, +20.28,                                         \
          +0.70, +0.40, -2.50                                           \
        });                                                             \
    thrust::device_vector<TYPE> input;                                  \
    imgdistort::gpu::testing::OriginalTensor<TYPE>(N, C, &input);       \
    /* Check C++ API */                                                 \
    thrust::device_vector<TYPE> output(input.size());                   \
    imgdistort::gpu::affine_nchw<TYPE, int>(                            \
        N, C, 16, 32, N, M.data().get(), input.data().get(), 32,        \
        output.data().get(), 32, 0);                                    \
    /* Check C API */                                                   \
    thrust::device_vector<TYPE> output2(input.size());                  \
    imgdistort_gpu_affine_nchw_##DESC(                                  \
        N, C, 16, 32, N, M.data().get(), input.data().get(), 32,        \
        output2.data().get(), 32, 0, nullptr);                          \
    thrust::host_vector<TYPE> input_cpu(input);                         \
    thrust::host_vector<TYPE> output_cpu(output);                       \
    thrust::host_vector<TYPE> output2_cpu(output2);                     \
                                                                        \
    std::vector<TYPE> tmp;                                              \
    imgdistort::cpu::testing::ExpectedGenericTensor<TYPE>(N, C, &tmp); \
    thrust::host_vector<TYPE> expected_output(tmp);                     \
    EXPECT_CONTAINER_NEAR(output_cpu, expected_output,                  \
                          static_cast<TYPE>(1));                        \
    EXPECT_CONTAINER_NEAR(output2_cpu, expected_output,                 \
                          static_cast<TYPE>(1));                        \
  }                                                                     \
                                                                        \
  INSTANTIATE_TEST_CASE_P(                                              \
      TestParameters_##DESC , AffineGPUTest_##DESC,                     \
      ::testing::Combine(                                               \
           ::testing::Values(1, 2),          /* N */                    \
           ::testing::Values(1, 2, 3, 4)))   /* C */

// REGISTER_TYPED_TESTS(s8,  int8_t); // Tests are not valid for int8_t
REGISTER_TYPED_TESTS(s16, int16_t);
REGISTER_TYPED_TESTS(s32, int32_t);
REGISTER_TYPED_TESTS(s64, int64_t);
REGISTER_TYPED_TESTS(u8,  uint8_t);
REGISTER_TYPED_TESTS(u16, uint16_t);
REGISTER_TYPED_TESTS(u32, uint32_t);
REGISTER_TYPED_TESTS(u64, uint64_t);
REGISTER_TYPED_TESTS(f32, float);
REGISTER_TYPED_TESTS(f64, double);

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
