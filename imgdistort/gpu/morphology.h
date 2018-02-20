#ifndef IMGDISTORT_GPU_MORPHOLOGY_H_
#define IMGDISTORT_GPU_MORPHOLOGY_H_

#include <cuda_runtime.h>
#include <imgdistort/gpu/defines.h>
#include <imgdistort/interpolation.h>
#include <imgdistort/logging.h>
#include <imgdistort/morphology_util.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cstdint>

// W,H Block size
#define MORPH_BLOCK_SIZE      16
// W,H Max Kernel sizes (up to 15 x 15 pixels)
#define MORPH_MAX_KERNEL_SIZE 15
// W,H Max apron area size
#define MORPH_MAX_APRON_SIZE  (MORPH_BLOCK_SIZE + MORPH_MAX_KERNEL_SIZE - 1)

#ifdef __cplusplus
namespace imgdistort {
namespace gpu {

namespace internal {

template <typename Int>
__global__
void kernel_compute_matrix_elems(const Int Mn, const Int* Ms, Int* M_elems) {
  for (Int n = thGx; n < Mn; n += NTGx) {
    M_elems[n] = Ms[2 * n + 0] * Ms[2 * n + 1];
  }
}

template <typename T, typename Int, typename Functor>
__global__
void kernel_morphology_nchw(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const Int* Ms, const Int* M_offset, const uint8_t* M,
    const T* src, const Int sp, T* dst, const Int dp) {
  const Int Bx = MORPH_BLOCK_SIZE * (blockIdx.x % DIV_UP(W, MORPH_BLOCK_SIZE));
  const Int By = MORPH_BLOCK_SIZE * blockIdx.y;

  __shared__ Int _T[3];
  __shared__ bool _M[MORPH_MAX_KERNEL_SIZE * MORPH_MAX_KERNEL_SIZE];
  __shared__ T _S[MORPH_MAX_APRON_SIZE * MORPH_MAX_APRON_SIZE];

  for (Int z = thGz; z < N * C; z += NTGz) {
    const Int c = z % C;
    const Int n = z / C;

    // Pointer to the current source and destination image/channel.
    const T* src_nc = src + n * C * H * sp + c * H * sp;
    T* dst_nc = dst + n * C * H * dp + c * H * dp;

    // Copy to shared memory the size of the structure kernel mask and the
    // offset for the current image (each block process an individual image).
    if (thBx == 0 && thBy == 0) {
      _T[0] = Ms[(n % Mn)    ];
      _T[1] = Ms[(n % Mn) + 1];
      _T[2] = M_offset[n % Mn];
    }
    __syncthreads();

    // Height, Width and offset of the structure kernel for the current image.
    const Int Mh = _T[0];
    const Int Mw = _T[1];
    const Int M_offset = _T[2];
    if (Mh < 0 || Mw < 0 || M_offset < 0) { asm("trap;"); }

    // Copy to shared memory the structure kernel mask for the current image.
    for (Int ki = thBy; ki < Mh; ki += blockDim.y) {
      for (Int kj = thBx; kj < Mw; kj += blockDim.x) {
        _M[ki * Mw + kj] = M[M_offset + ki * Mw + kj];
      }
    }

    // Copy to shared memory the source image.
    const Int Aw = blockDim.x + Mw - 1, Ah = blockDim.y + Mh - 1;
    for (Int ay = thBy; ay < Ah; ay += blockDim.y) {
      for (Int ax = thBx; ax < Aw; ax += blockDim.x) {
        const Int sy = min(max(By + ay - Mh / 2, (Int)0), H - 1);
        const Int sx = min(max(Bx + ax - Mw / 2, (Int)0), W - 1);
        _S[ay * Aw + ax] = pixv(src_nc, sp, sy, sx);
      }
    }
    __syncthreads();

    // Compute output pixel value
    const Int x = Bx + thBx;
    const Int y = By + thBy;
    if (x < W && y < H) {
      bool init = false;
      T tmp = 0; // pixv(_S, Aw, thBy + Mh / 2, thBx + Mw / 2);
      for (Int ki = 0; ki < Mh; ++ki) {
        for (Int kj = 0; kj < Mw; ++kj) {
          const Int ay = thBy + ki, ax = thBx + kj;
          const Int sy = min(max(By + ay - Mh / 2, (Int)0), H - 1);
          const Int sx = min(max(Bx + ax - Mw / 2, (Int)0), W - 1);
          if (_M[ki * Mw + kj] != 0) {
            if (!init) { tmp = pixv(_S, Aw, ay, ax); init = true; }
            else { tmp = Functor::f(tmp, pixv(_S, Aw, ay, ax)); }
          }
        }
      }
      pixv(dst_nc, dp, y, x) = tmp;
    }
  }
}

template <typename T, typename Int, typename Functor>
void morphology_nchw(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const Int* Ms, const uint8_t* M,
    const T* src, const Int sp, T* dst, const Int dp,
    cudaStream_t stream) {
  // Check image sizes
  CHECK_GT(N, 0); CHECK_GT(C, 0); CHECK_GT(H, 0); CHECK_GT(W, 0);
  // Check transformation kernels
  CHECK_GT(Mn, 0); CHECK_NOTNULL(Ms); CHECK_NOTNULL(M);
  // Check source and dest images and pitches
  CHECK_NOTNULL(src); CHECK_GT(sp, 0);
  CHECK_NOTNULL(dst); CHECK_GT(dp, 0);

  thrust::device_vector<Int> M_offset(Mn);
  // compute the total number of elements in each structure matrix
  kernel_compute_matrix_elems<Int>
      <<<NUM_BLOCKS(Mn, 512), 512, 0, stream>>>(Mn, Ms, M_offset.data().get());
  // compute the offset of each matrix
  thrust::exclusive_scan(
      thrust::cuda::par.on(stream), M_offset.begin(), M_offset.end(),
      M_offset.begin());
  const dim3 block_size(MORPH_BLOCK_SIZE, MORPH_BLOCK_SIZE, 1);
  const dim3 grid_size(NUM_BLOCKS(W, MORPH_BLOCK_SIZE),
                       NUM_BLOCKS(H, MORPH_BLOCK_SIZE),
                       NUM_BLOCKS(N * C, 1));
  const Int* M_offset_ptr = M_offset.data().get();
  kernel_morphology_nchw<T, Int, Functor>
      <<<grid_size, block_size, 0, stream>>>(
          N, C, H, W, Mn, Ms, M_offset_ptr, M, src, sp, dst, dp);
  if (stream == 0) { CHECK_LAST_CUDA_CALL(); }
}

}  // namespace internal

template <typename T, typename Int>
void dilate_nchw(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const Int* Ms, const uint8_t* M,
    const T* src, const Int sp, T* dst, const Int dp, cudaStream_t stream = nullptr) {
  using ::imgdistort::internal::DilateFunctor;
  internal::morphology_nchw<T, Int, DilateFunctor<T>>(
      N, C, H, W, Mn, Ms, M, src, sp, dst, dp, stream);
}

template <typename T, typename Int>
void erode_nchw(
    const Int N, const Int C, const Int H, const Int W,
    const Int Mn, const Int* Ms, const uint8_t* M,
    const T* src, const Int sp, T* dst, const Int dp, cudaStream_t stream = nullptr) {
  using ::imgdistort::internal::ErodeFunctor;
  internal::morphology_nchw<T, Int, ErodeFunctor<T>>(
      N, C, H, W, Mn, Ms, M, src, sp, dst, dp, stream);
}

}  // namespace gpu
}  // namespace imgdistort
#endif  // __cplusplus

// C bindings
#define DECLARE_BINDING(TYPE, SNAME)                              \
  EXTERN_C void imgdistort_gpu_dilate_nchw_##SNAME(               \
      const int N, const int C, const int H, const int W,         \
      const int Mn, const int* Ms, const uint8_t* M,              \
      const TYPE* src, const int sp, TYPE* dst, const int dp,     \
      cudaStream_t stream);                                       \
                                                                  \
  EXTERN_C void imgdistort_gpu_erode_nchw_##SNAME(                \
      const int N, const int C, const int H, const int W,         \
      const int Mn, const int* Ms, const uint8_t* M,              \
      const TYPE* src, const int sp, TYPE* dst, const int dp,     \
      cudaStream_t stream)

DECLARE_BINDING(float, f32);
DECLARE_BINDING(double, f64);
DECLARE_BINDING(int8_t, s8);
DECLARE_BINDING(int16_t, s16);
DECLARE_BINDING(int32_t, s32);
DECLARE_BINDING(int64_t, s64);
DECLARE_BINDING(uint8_t, u8);
DECLARE_BINDING(uint16_t, u16);
DECLARE_BINDING(uint32_t, u32);
DECLARE_BINDING(uint64_t, u64);
#undef DECLARE_BINDING

#endif  // IMGDISTORT_GPU_MORPHOLOGY_H_
