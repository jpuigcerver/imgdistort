#include <gtest/gtest.h>  // IWYU pragma: keep
#include <imgdistort/gpu/morphology.h>
#include <imgdistort/testing/morphology_base.h>
#include <imgdistort/testing/device_allocator_gpu.h>

#include <cstdint>
#include <vector>

namespace imgdistort {
namespace testing {
namespace gpu {

// Wrapper around the C++ interface
template <typename T>
class MorphologyGpuCppInterface {
 public:
  typedef T Type;
  typedef DeviceAllocator<GPU> Allocator;

  static void Dilate(const int N, const int C, const int H, const int W,
                     const int Mn, const int *Ms, const uint8_t *M,
                     const T *src, const int sp, T *dst, const int dp) {
    ::imgdistort::gpu::dilate_nchw(N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
  }

  static void Erode(const int N, const int C, const int H, const int W,
                    const int Mn, const int *Ms, const uint8_t *M,
                    const T *src, const int sp, T *dst, const int dp) {
    ::imgdistort::gpu::erode_nchw(N, C, H, W, Mn, Ms, M, src, sp, dst, dp);
  }
};

// Wrapper around the C interface
template <typename T>
class CInterface {
 public:
  typedef T Type;
  typedef DeviceAllocator<GPU> Allocator;

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
    imgdistort_gpu_dilate_nchw_##DESC(                                \
        N, C, H, W, Mn, Ms, M, src, sp, dst, dp, nullptr);            \
  }                                                                   \
  template <>                                                         \
  void CInterface<TYPE>::Erode(                                       \
      const int N, const int C, const int H, const int W,             \
      const int Mn, const int *Ms, const uint8_t *M,                  \
      const TYPE *src, const int sp, TYPE *dst, const int dp) {       \
    imgdistort_gpu_erode_nchw_##DESC(                                 \
        N, C, H, W, Mn, Ms, M, src, sp, dst, dp, nullptr);            \
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

}  // namespace gpu

typedef ::testing::Types<
    gpu::MorphologyGpuCppInterface<int16_t>,
    gpu::MorphologyGpuCppInterface<int32_t>,
    gpu::MorphologyGpuCppInterface<int64_t>,
    gpu::MorphologyGpuCppInterface<uint8_t>,
    gpu::MorphologyGpuCppInterface<uint16_t>,
    gpu::MorphologyGpuCppInterface<uint32_t>,
    gpu::MorphologyGpuCppInterface<uint64_t>,
    gpu::MorphologyGpuCppInterface<float>,
    gpu::MorphologyGpuCppInterface<double> > CppTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Gpu, MorphologyTest, CppTypes);

typedef ::testing::Types<
    gpu::CInterface<int16_t>,
    gpu::CInterface<int32_t>,
    gpu::CInterface<int64_t>,
    gpu::CInterface<uint8_t>,
    gpu::CInterface<uint16_t>,
    gpu::CInterface<uint32_t>,
    gpu::CInterface<uint64_t>,
    gpu::CInterface<float>,
    gpu::CInterface<double> > CTypes;
INSTANTIATE_TYPED_TEST_CASE_P(GpuC, MorphologyTest, CTypes);
}  // namespace testing
}  // namespace imgdistort
