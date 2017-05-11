#include <cstdint>
#include <vector>

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>  // IWYU pragma: keep
#include <imgdistort/affine_cpu.h>

#include "base_test.h"
#include "affine_base_test_cpu.h"

typedef std::tuple<int, int> TestArgs;

#define REGISTER_TYPED_TESTS(DESC, TYPE)                                \
  class AffineCPUTest_##DESC :                                          \
      public ::testing::TestWithParam<TestArgs> {};                     \
                                                                        \
  TEST_P(AffineCPUTest_##DESC, Idempotent) {                            \
    const TestArgs param = GetParam();                                  \
    const int N = std::get<0>(param);                                   \
    const int C = std::get<1>(param);                                   \
    const double M[] = {                                                \
      1.0, 0.0, 0.0,                                                    \
      0.0, 1.0, 0.0                                                     \
    };                                                                  \
    std::vector<TYPE> input;                                            \
    imgdistort::testing::cpu::OriginalTensor<TYPE>(N, C, &input);       \
    /* Check C++ API */                                                 \
    std::vector<TYPE> output(input.size());                             \
    imgdistort::cpu::affine_nchw(                                       \
        N, C, 16, 32, M, 1, input.data(), 32, output.data(), 32);       \
    /* Check C API */                                                   \
    std::vector<TYPE> output2(input.size());                            \
    imgdistort_cpu_affine_nchw_##DESC(                                  \
        N, C, 16, 32, M, 1, input.data(), 32, output2.data(), 32);      \
    EXPECT_THAT(output, ::testing::ElementsAreArray(input));            \
    EXPECT_THAT(output2, ::testing::ElementsAreArray(input));           \
  }                                                                     \
                                                                        \
  TEST_P(AffineCPUTest_##DESC, Generic) {                               \
    const TestArgs param = GetParam();                                  \
    const int N = std::get<0>(param);                                   \
    const int C = std::get<1>(param);                                   \
    const double M[] = {                                                \
      /* Matrix 1 */                                                    \
      +0.76, +0.83, -0.18,                                              \
      -0.05, +0.78, +1.18,                                              \
      /* Matrix 2 */                                                    \
      -1.40, +0.70, +20.28,                                             \
      +0.70, +0.40, -2.50                                               \
    };                                                                  \
    std::vector<TYPE> input;                                            \
    imgdistort::testing::cpu::OriginalTensor<TYPE>(N, C, &input);       \
    /* Check C++ API */                                                 \
    std::vector<TYPE> output(input.size());                             \
    imgdistort::cpu::affine_nchw(                                       \
        N, C, 16, 32, M, N, input.data(), 32, output.data(), 32);       \
    /* Check C API */                                                   \
    std::vector<TYPE> output2(input.size());                            \
    imgdistort_cpu_affine_nchw_##DESC(                                  \
        N, C, 16, 32, M, N, input.data(), 32, output2.data(), 32);      \
                                                                        \
    std::vector<TYPE> expected_output;                                  \
    imgdistort::testing::cpu::ExpectedGenericTensor<TYPE>(              \
        N, C, &expected_output);                                        \
    EXPECT_CONTAINER_NEAR(output, expected_output,                      \
                          static_cast<TYPE>(1));                        \
    EXPECT_CONTAINER_NEAR(output2, expected_output,                     \
                          static_cast<TYPE>(1));                        \
  }                                                                     \
                                                                        \
  INSTANTIATE_TEST_CASE_P(                                              \
      TestParameters_##DESC , AffineCPUTest_##DESC,                     \
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
