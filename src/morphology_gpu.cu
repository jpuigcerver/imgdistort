#include <imgdistort/morphology_gpu.h>
#include <imgdistort/defines.h>

#include <glog/logging.h>
#include <imgdistort/interpolation.h>

// W,H Block size
#define BLOCK_SIZE      16
// W,H Max Kernel sizes (up to 15 x 15 pixels)
#define MAX_KERNEL_SIZE 15
// W,H Max apron area size
#define MAX_APRON_SIZE  (BLOCK_SIZE + MAX_KERNEL_SIZE - 1)

namespace imgdistort {
namespace gpu {

#define DEFINE_IMPLEMENTATION(T)                                        \
  template <>                                                           \
  void dilate_nchw<T>(                                                  \
      const int N, const int C, const int H, const int W,               \
      const uint8_t* M, const int* Ms, const int Mn,                    \
      const T* src, const int sp, T* dst, const int dp,                 \
      cudaStream_t stream) {                                            \
    constexpr T padv = std::numeric_limits<T>::lowest();                \
    morphology_nchw<T, MaxFunctor<T>>(                                  \
        N, C, H, W, M, Ms, Mn, src, sp, dst, dp, padv, stream);         \
  }                                                                     \
  template <>                                                           \
  void erode_nchw<T>(                                                   \
      const int N, const int C, const int H, const int W,               \
      const uint8_t* M, const int* Ms, const int Mn,                    \
      const T* src, const int sp, T* dst, const int dp,                 \
      cudaStream_t stream) {                                            \
    constexpr T padv = std::numeric_limits<T>::max();                   \
    morphology_nchw<T, MinFunctor<T>>(                                  \
        N, C, H, W, M, Ms, Mn, src, sp, dst, dp, padv, stream);         \
  }

template <typename T>
struct MaxFunctor {
  __host__ __device__
  static inline const T& f(const T& a, const T& b) {
    return a > b ? a : b;
  }
};

template <typename T>
struct MinFunctor {
  __host__ __device__
  static inline const T& f(const T& a, const T& b) {
    return a < b ? a : b;
  }
};

template <typename T, class F>
__global__
static void kernel_morphology_nchw(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int Mn, const int Mh, const int Mw,
    const T* src, const int sp, T* dst, const int dp, const T padv) {
  const int Bx = BLOCK_SIZE * (blockIdx.x % DIV_UP(W, BLOCK_SIZE));
  const int By = BLOCK_SIZE * blockIdx.y;
  const int n = blockIdx.z;
  const int c = blockIdx.x / DIV_UP(W, BLOCK_SIZE);
  const int x = Bx + threadIdx.x;
  const int y = By + threadIdx.y;
  const int offset_S = (n * C + c) * H * sp;
  const int offset_D = (n * C + c) * H * dp;
  const int offset_M = (n % Mn) * Mh * Mw;

  // Copy structure kernel mask to shared memory.
  __shared__ bool _M[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
  if (threadIdx.x < Mw && threadIdx.y < Mh) {
    _M[threadIdx.x + threadIdx.y * Mw] =
        M[offset_M + threadIdx.y * Mw + threadIdx.x];
  }

  // Copy source image to shared memory
  __shared__ T _S[MAX_APRON_SIZE * MAX_APRON_SIZE];
  const int Aw = BLOCK_SIZE + Mw - 1, Ah = BLOCK_SIZE + Mh - 1;
  const int L = DIV_UP(Aw * Ah, BLOCK_SIZE * BLOCK_SIZE);
  for (int l = 0, j = L * thBi; l < L && j < Aw * Ah; ++l, ++j) {
    const int ax = j % Aw,           ay = j / Aw;
    const int sx = Bx + ax - Mw / 2, sy = By + ay - Mh / 2;
    _S[j] = pixv(src + offset_S, sp, H, W, sy, sx, padv);
  }
  __syncthreads();

  // Compute output pixel value
  if (x >= W || y >= H) return;
  T tmp = padv;
  for (int ki = 0; ki < Mh; ++ki) {
    for (int kj = 0; kj < Mw; ++kj) {
      if (_M[(Mn > 1 ? n : 0) * Mh * Mw + ki * Mw + kj] != 0) {
        tmp = F::f(tmp, pixv(_S, Aw, threadIdx.y + ki, threadIdx.x + kj));
      }
    }
  }
  pixv(dst + offset_D, dp, y, x) = tmp;
}

template <typename T, class F>
static inline void morphology_nchw(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const T* src, const int sp, T* dst, const int dp, const T padv,
    cudaStream_t stream) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0);  CHECK_GT(W, 0);
  // Check transformation kernels
  CHECK_NOTNULL(M); CHECK_GT(Mn, 0); CHECK(Mn == 1 || Mn == N);
  CHECK_NOTNULL(Ms);
  // Check source and dest images and pitches
  CHECK_NOTNULL(src); CHECK_GT(sp, 0);
  CHECK_NOTNULL(dst); CHECK_GT(dp, 0);

  const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
  const dim3 grid_size(C * DIV_UP(W, BLOCK_SIZE), DIV_UP(H, BLOCK_SIZE), N);

  bool all_kernels_same_size = true;
  for (int i = 1; i < Mn && all_kernels_same_size; ++i) {
    all_kernels_same_size = (Ms[2 * i] == Ms[0] && Ms[2 * i + 1] == Ms[1]);
  }

  if (all_kernels_same_size) {
    // If all morphology kernels have the same size, we can launch a single
    // CUDA kernel to process all images.
    const int Mh = Ms[0];
    const int Mw = Ms[1];
    CHECK_GT(Mh, 0); CHECK_GT(Mw, 0);
    // GPU implementation restrictions
    CHECK_LE(Mh, MAX_KERNEL_SIZE)
      << "GPU implementation cannot handle a morphology operation with "
      << "a kernel size larger than " << MAX_KERNEL_SIZE << " pixels";
    CHECK_LE(Mw, MAX_KERNEL_SIZE)
      << "GPU implementation cannot handle a morphology operation with "
      << "a kernel size larger than " << MAX_KERNEL_SIZE << " pixels";
    kernel_morphology_nchw<T, F><<<grid_size, block_size, 0, stream>>>
        (N, C, H, W, M, Mn, Mh, Mw, src, sp, dst, dp, padv);
    CHECK_LAST_CUDA_CALL();
    if (stream == 0) {
      CHECK_CUDA_CALL(cudaDeviceSynchronize());
    }
  } else {
    // Each image uses a kernel with a different size. We need to launch a
    // kernel for each image. (All kernels are launched into parallel
    // streams).
    CHECK_CUDA_CALL(cudaStreamSynchronize(stream));
    cudaStream_t* streams = new cudaStream_t[N];
    size_t offset_M = 0;
    size_t offset_S = 0;
    size_t offset_D = 0;
    for (int n = 0; n < N; ++n) {
      CHECK_CUDA_CALL(cudaStreamCreate(streams + n));
      const int Mh = Ms[2 * (n % Mn) + 0];
      const int Mw = Ms[2 * (n % Mn) + 1];
      CHECK_GT(Mh, 0); CHECK_GT(Mw, 0);
      // GPU implementation restrictions
      CHECK_LE(Mh, MAX_KERNEL_SIZE)
        << "GPU implementation cannot handle a morphology operation with "
        << "a kernel size larger than " << MAX_KERNEL_SIZE << " pixels";
      CHECK_LE(Mw, MAX_KERNEL_SIZE)
        << "GPU implementation cannot handle a morphology operation with "
        << "a kernel size larger than " << MAX_KERNEL_SIZE << " pixels";
      kernel_morphology_nchw<T, F><<<grid_size, block_size, 0, stream>>>(
          1, C, H, W, M + offset_M, 1, Mh, Mw, src + offset_S, sp,
          dst + offset_D, dp, padv);
      CHECK_LAST_CUDA_CALL();
      offset_M += Mh * Mw;
      offset_S += H * sp;
      offset_D += H * dp;
    }
    for (int n = 0; n < N; ++n) {
      CHECK_CUDA_CALL(cudaStreamSynchronize(streams[n]));
    }
  }
}

DEFINE_IMPLEMENTATION(int8_t)
DEFINE_IMPLEMENTATION(int16_t)
DEFINE_IMPLEMENTATION(int32_t)
DEFINE_IMPLEMENTATION(int64_t)
DEFINE_IMPLEMENTATION(uint8_t)
DEFINE_IMPLEMENTATION(uint16_t)
DEFINE_IMPLEMENTATION(uint32_t)
DEFINE_IMPLEMENTATION(uint64_t)
DEFINE_IMPLEMENTATION(float)
DEFINE_IMPLEMENTATION(double)

}  // namespace gpu
}  // namespace imgdistort

#define DEFINE_C_FUNCTION(DESC, TYPE)                               \
  extern "C" void imgdistort_gpu_dilate_nchw_ ## DESC  (            \
      const int N, const int C, const int H, const int W,           \
      const uint8_t* M, const int* Ms, const int Mn,                \
      const TYPE* src, const int sp, TYPE* dst, const int dp,       \
      cudaStream_t stream) {                                        \
    imgdistort::gpu::dilate_nchw<TYPE>(                             \
        N, C, H, W, M, Ms, Mn, src, sp, dst, dp, stream);           \
  }                                                                 \
  extern "C" void imgdistort_gpu_erode_nchw_ ## DESC  (             \
      const int N, const int C, const int H, const int W,           \
      const uint8_t* M, const int* Ms, const int Mn,                \
      const TYPE* src, const int sp, TYPE* dst, const int dp,       \
      cudaStream_t stream) {                                        \
    imgdistort::gpu::erode_nchw<TYPE>(                              \
        N, C, H, W, M, Ms, Mn, src, sp, dst, dp, stream);           \
  }

DEFINE_C_FUNCTION(s8,  int8_t)
DEFINE_C_FUNCTION(s16, int16_t)
DEFINE_C_FUNCTION(s32, int32_t)
DEFINE_C_FUNCTION(s64, int64_t)
DEFINE_C_FUNCTION(u8,  uint8_t)
DEFINE_C_FUNCTION(u16, uint16_t)
DEFINE_C_FUNCTION(u32, uint32_t)
DEFINE_C_FUNCTION(u64, uint64_t)
DEFINE_C_FUNCTION(f32, float)
DEFINE_C_FUNCTION(f64, double)