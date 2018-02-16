#include <imgdistort/gpu/affine.h>

// C bindings
#define DEFINE_BINDING(TYPE, SNAME)                                     \
  EXTERN_C void imgdistort_gpu_affine_nchw_##SNAME(                     \
      const int N, const int C, const int H, const int W,               \
      const int Mn, const double* M,                                    \
      const TYPE* src, const int sp, TYPE* dst, const int dp,           \
      const TYPE border_value, cudaStream_t stream) {                   \
    ::imgdistort::gpu::affine_nchw<TYPE, int>(                          \
         N, C, H, W, Mn, M, src, sp, dst, dp, border_value, stream);    \
  }

DEFINE_BINDING(float, f32);
DEFINE_BINDING(double, f64);
DEFINE_BINDING(int8_t, s8);
DEFINE_BINDING(int16_t, s16);
DEFINE_BINDING(int32_t, s32);
DEFINE_BINDING(int64_t, s64);
DEFINE_BINDING(uint8_t, u8);
DEFINE_BINDING(uint16_t, u16);
DEFINE_BINDING(uint32_t, u32);
DEFINE_BINDING(uint64_t, u64);
