#include <gtest/gtest.h>  // IWYU pragma: keep
#include <imgdistort/cpu/morphology.h>
#include <imgdistort/testing/morphology_base.h>
#include <imgdistort/testing/device_allocator_cpu.h>

#include <cstdint>
#include <vector>

namespace imgdistort {
namespace testing {
namespace cpu {

// Wrapper around the C++ interface
template <typename T>
class CppInterface {
 public:
  typedef T Type;
  typedef DeviceAllocator<CPU> Allocator;

  static void Dilate(const int N, const int C, const int H, const int W,
                     const int Mn, const int *Ms, const uint8_t *M,
                     const T *src, const int sp, T *dst, const int dp) {
    ::imgdistort::cpu::dilate_nchw<T, int>(N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
  }

  static void Erode(const int N, const int C, const int H, const int W,
                    const int Mn, const int *Ms, const uint8_t *M,
                    const T *src, const int sp, T *dst, const int dp) {
    ::imgdistort::cpu::erode_nchw<T, int>(N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
  }
};

// Wrapper around the C interface
template <typename T>
class CInterface {
 public:
  typedef T Type;
  typedef DeviceAllocator<CPU> Allocator;

  static void Dilate(const int N, const int C, const int H, const int W,
                     const int Mn, const int *Ms, const uint8_t *M,
                     const T *src, const int sp, T *dst, const int dp);

  static void Erode(const int N, const int C, const int H, const int W,
                    const int Mn, const int *Ms, const uint8_t *M,
                    const T *src, const int sp, T *dst, const int dp);
};

#define DEFINE_C_SPECIALIZATION(DESC, TYPE)                           \
  template <>                                                         \
  void CInterface<TYPE>::Dilate(                                      \
      const int N, const int C, const int H, const int W,             \
      const int Mn, const int *Ms, const uint8_t *M,                  \
      const TYPE *src, const int sp, TYPE *dst, const int dp) {       \
    imgdistort_cpu_dilate_nchw_##DESC(                                \
        N, C, H, W, Mn, Ms, M, src, sp, dst, dp);                     \
  }                                                                   \
                                                                      \
  template <>                                                         \
  void CInterface<TYPE>::Erode(                                       \
      const int N, const int C, const int H, const int W,             \
      const int Mn, const int *Ms, const uint8_t *M,                  \
      const TYPE *src, const int sp, TYPE *dst, const int dp) {       \
    imgdistort_cpu_erode_nchw_##DESC(                                 \
        N, C, H, W, Mn, Ms, M, src, sp, dst, dp);                     \
  }

DEFINE_C_SPECIALIZATION(s16, int16_t)
DEFINE_C_SPECIALIZATION(s32, int32_t)
DEFINE_C_SPECIALIZATION(s64, int64_t)
DEFINE_C_SPECIALIZATION(u8,  uint8_t)
DEFINE_C_SPECIALIZATION(u16, uint16_t)
DEFINE_C_SPECIALIZATION(u32, uint32_t)
DEFINE_C_SPECIALIZATION(u64, uint64_t)
DEFINE_C_SPECIALIZATION(f32, float)
DEFINE_C_SPECIALIZATION(f64, double)

}  // namespace cpu

typedef ::testing::Types<
  cpu::CppInterface<int16_t>,
  cpu::CppInterface<int32_t>,
  cpu::CppInterface<int64_t>,
  cpu::CppInterface<uint8_t>,
  cpu::CppInterface<uint16_t>,
  cpu::CppInterface<uint32_t>,
  cpu::CppInterface<uint64_t>,
  cpu::CppInterface<float>,
  cpu::CppInterface<double> > CppTypes;
INSTANTIATE_TYPED_TEST_CASE_P(CpuCpp, MorphologyTest, CppTypes);

typedef ::testing::Types<
  cpu::CInterface<int16_t>,
  cpu::CInterface<int32_t>,
  cpu::CInterface<int64_t>,
  cpu::CInterface<uint8_t>,
  cpu::CInterface<uint16_t>,
  cpu::CInterface<uint32_t>,
  cpu::CInterface<uint64_t>,
  cpu::CInterface<float>,
  cpu::CInterface<double> > CTypes;
INSTANTIATE_TYPED_TEST_CASE_P(CpuC, MorphologyTest, CTypes);

}  // namespace testing
}  // namespace imgdistort
