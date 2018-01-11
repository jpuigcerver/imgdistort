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
    morphology_nchw<T, MaxFunctor<T>>(                                  \
        N, C, H, W, M, Ms, Mn, src, sp, dst, dp, stream);               \
  }                                                                     \
  template <>                                                           \
  void erode_nchw<T>(                                                   \
      const int N, const int C, const int H, const int W,               \
      const uint8_t* M, const int* Ms, const int Mn,                    \
      const T* src, const int sp, T* dst, const int dp,                 \
      cudaStream_t stream) {                                            \
    morphology_nchw<T, MinFunctor<T>>(                                  \
        N, C, H, W, M, Ms, Mn, src, sp, dst, dp, stream);               \
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
    const T* src, const int sp, T* dst, const int dp) {
  const int Bx = BLOCK_SIZE * (blockIdx.x % DIV_UP(W, BLOCK_SIZE));
  const int By = BLOCK_SIZE * blockIdx.y;
  const int n = blockIdx.z;
  const int c = blockIdx.x / DIV_UP(W, BLOCK_SIZE);
  const int offset_S = (n * C + c) * H * sp;
  const int offset_D = (n * C + c) * H * dp;
  const int offset_M = (n % Mn) * Mh * Mw;

  // Copy structure kernel mask to shared memory.
  __shared__ bool _M[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
  for (int ki = threadIdx.y; ki < Mh; ki += blockDim.y) {
    for (int kj = threadIdx.x; kj < Mw; kj += blockDim.x) {
      _M[ki * Mw + kj] = M[offset_M + ki * Mw + kj];
    }
  }

  // Copy source image to shared memory
  __shared__ T _S[MAX_APRON_SIZE * MAX_APRON_SIZE];
  const int Aw = blockDim.x + Mw - 1, Ah = blockDim.y + Mh - 1;
  for (int ay = threadIdx.y; ay < Ah; ay += blockDim.y) {
    for (int ax = threadIdx.x; ax < Aw; ax += blockDim.x) {
      const int sx = Bx + ax - Mw / 2, sy = By + ay - Mh / 2;
      if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
        _S[ay * Aw + ax] = pixv(src + offset_S, sp, sy, sx);
      }
    }
  }
  __syncthreads();

  // Compute output pixel value
  const int x = Bx + threadIdx.x;
  const int y = By + threadIdx.y;
  if (x < W && y < H) {
    T tmp = pixv(_S, Aw, threadIdx.y + Mh / 2, threadIdx.x + Mw / 2);
    for (int ki = 0; ki < Mh; ++ki) {
      for (int kj = 0; kj < Mw; ++kj) {
        const int ay = threadIdx.y + ki, ax = threadIdx.x + kj;
        const int sy = By + ay - Mh / 2, sx = Bx + ax - Mw / 2;
        if (sx >= 0 && sx < W && sy >= 0 && sy < H && _M[ki * Mw + kj]) {
          tmp = F::f(tmp, pixv(_S, Aw, ay, ax));
        }
      }
    }
    pixv(dst + offset_D, dp, y, x) = tmp;
  }
}

template <typename T, class F>
static inline void morphology_nchw(
    const int N, const int C, const int H, const int W,
    const uint8_t* M, const int* Ms, const int Mn,
    const T* src, const int sp, T* dst, const int dp,
    cudaStream_t stream) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0);  CHECK_GT(W, 0);
  // Check transformation kernels
  CHECK_NOTNULL(M);
  CHECK_NOTNULL(Ms);
  CHECK_GT(Mn, 0); CHECK(Mn == 1 || Mn == N);
  // Check source and dest images and pitches
  CHECK_NOTNULL(src); CHECK_GT(sp, 0);
  CHECK_NOTNULL(dst); CHECK_GT(dp, 0);
  const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);

  bool all_kernels_same_size = true;
  for (int i = 0; i < Mn && all_kernels_same_size; ++i) {
    const int Mh = Ms[2 * i + 0];
    const int Mw = Ms[2 * i + 1];
    CHECK_GT(Mh, 0); CHECK_GT(Mw, 0);
    // GPU implementation restrictions.
    CHECK_LE(Mh, MAX_KERNEL_SIZE)
      << "GPU implementation cannot handle a morphology operation with "
      << "a kernel size larger than " << MAX_KERNEL_SIZE << " pixels";
    CHECK_LE(Mw, MAX_KERNEL_SIZE)
      << "GPU implementation cannot handle a morphology operation with "
      << "a kernel size larger than " << MAX_KERNEL_SIZE << " pixels";
    all_kernels_same_size = (Mh == Ms[0] && Mw == Ms[1]);
  }

  if (all_kernels_same_size) {
    // If all morphology kernels have the same size, we can launch a single
    // CUDA kernel to process all images.
    const int Mh = Ms[0];
    const int Mw = Ms[1];
    const dim3 grid_size(C * DIV_UP(W, BLOCK_SIZE), DIV_UP(H, BLOCK_SIZE), N);
    kernel_morphology_nchw<T, F><<<grid_size, block_size, 0, stream>>>
        (N, C, H, W, M, Mn, Mh, Mw, src, sp, dst, dp);
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
    const dim3 grid_size(C * DIV_UP(W, BLOCK_SIZE), DIV_UP(H, BLOCK_SIZE), 1);
    for (int n = 0; n < N; ++n) {
      CHECK_CUDA_CALL(cudaStreamCreate(&streams[n]));
      const int Mh = Ms[2 * (n % Mn) + 0];
      const int Mw = Ms[2 * (n % Mn) + 1];
      kernel_morphology_nchw<T, F><<<grid_size, block_size, 0, streams[n]>>>(
          1, C, H, W, M + offset_M, 1, Mh, Mw, src + offset_S, sp,
          dst + offset_D, dp);
      offset_M += Mh * Mw;
      offset_S += H * sp;
      offset_D += H * dp;
    }
    for (int n = 0; n < N; ++n) {
      CHECK_CUDA_CALL(cudaStreamSynchronize(streams[n]));
      CHECK_CUDA_CALL(cudaStreamDestroy(streams[n]));
    }
    delete [] streams;
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
