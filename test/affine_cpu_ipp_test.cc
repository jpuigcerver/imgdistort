#include <gtest/gtest.h>
#include <imgdistort/affine_cpu_ipp.h>

template <typename T>
class AffineIPPTest : public ::testing::Test {
};
TYPED_TEST_CASE_P(AffineIPPTest);

TYPED_TEST_P(AffineIPPTest, Idempotent) {
}

TYPED_TEST_P(AffineIPPTest, Rotate) {
}

TYPED_TEST_P(AffineIPPTest, Scale) {
}

REGISTER_TYPED_TEST_CASE_P(AffineIPPTest, Idempotent, Rotate, Scale);


INSTANTIATE_TYPED_TEST_CASE_P(Float, AffineIPPTest, float);
INSTANTIATE_TYPED_TEST_CASE_P(Double, AffineIPPTest, double);
