#include <cstdint>
#include <vector>

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>  // IWYU pragma: keep
#include <imgdistort/morphology_cpu.h>

typedef std::tuple<int, int> TestArgs;

#define REGISTER_TYPED_TESTS(DESC, TYPE)                                \
  class MorphologyCPUTest_##DESC :                                      \
      public ::testing::TestWithParam<TestArgs> {};                     \
                                                                        \
  TEST_P(MorphologyCPUTest_##DESC, Test) {                              \
                                                                        \
  }                                                                     \
  INSTANTIATE_TEST_CASE_P(                                              \
      TestParameters_##DESC , MorphologyCPUTest_##DESC,                 \
      ::testing::Combine(                                               \
           ::testing::Values(1, 2),          /* N */                    \
           ::testing::Values(1, 2, 3, 4)))   /* C */

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
