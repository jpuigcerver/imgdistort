#ifndef IMGDISTORT_SRC_UTILS_H_
#define IMGDISTORT_SRC_UTILS_H_

#include <cuda_runtime.h>
#include <glog/logging.h>

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

#define CHECK_CUDA_CALL(err) {                                          \
    const cudaError_t e = (err);                                        \
    CHECK_EQ(e, cudaSuccess) << "CUDA error : " << e << " ("            \
                             << cudaGetErrorString(e)  << ")";          \
  }

#define CHECK_LAST_CUDA_CALL CHECK_CUDA_CALL(cudaGetLastError())

#define coord_in(x, y, w, h) \
  (((x) < 0 || (y) < 0 || (x) >= (w) || (y) >= (h)) ? 0 : 1)
#define pixv(src, x, y, w, h) ((src)[(y) * (w) + (x)])
#define pixv_pad(src, x, y, w, h, v)                                  \
  (((x) < 0||(y) < 0||(x) >= (w)||(y) >= (h)) ? (v) : ((src)[(y) * (w) + (x)]))

template <typename T>
__host__ __device__
float blinterp(const T* src, float x, float y, int w, int h) {
  const int x_i = static_cast<int>(x);
  const int y_i = static_cast<int>(y);
  const float a = x - x_i;
  const float b = y - y_i;
  float n = 0;
  float v = 0.0f;
  if (coord_in(x_i    , y_i    , w, h)) {
    v += (1 - a) * (1 - b) * pixv(src, x_i    , y_i    , w, h);
    n += (1 - a) * (1 - b);
  }
  if (coord_in(x_i + 1, y_i    , w, h)) {
    v += (    a) * (1 - b) * pixv(src, x_i + 1, y_i    , w, h);
    n += (    a) * (1 - b);
  }
  if (coord_in(x_i    , y_i + 1, w, h)) {
    v += (1 - a) * (    b) * pixv(src, x_i    , y_i + 1, w, h);
    n += (1 - a) * (    b);

  }
  if (coord_in(x_i + 1, y_i + 1, w, h)) {
    v += (    a) * (    b) * pixv(src, x_i + 1, y_i + 1, w, h);
    n += (    a) * (    b);
  }
  return n > 0.0f ? v / n : 0.0f;
}

// Some definitions that only make sense when using CUDACC
#ifdef __CUDACC__
// Thread IDs within a block
#define thBx (threadIdx.x)
#define thBy (threadIdx.y)
#define thBz (threadIdx.z)
#define thBi (                                          \
    threadIdx.x +                                       \
    threadIdx.y * blockDim.x +                          \
    threadIdx.z * blockDim.x * blockDim.y)
// Number of threads in a block
#define NTB (blockDim.x * blockDim.y * blockDim.z)

// Thread IDs within the grid (global IDs)
#define thGx (threadIdx.x + blockIdx.x * blockDim.x)
#define thGy (threadIdx.y + blockIdx.y * blockDim.y)
#define thGz (threadIdx.z + blockIdx.z * blockDim.z)
#define thGi (                                                          \
    threadIdx.x +                                                       \
    threadIdx.y * blockDim.x +                                          \
    threadIdx.z * blockDim.x * blockDim.y +                             \
    (blockIdx.x +                                                       \
     blockIdx.y * gridDim.x +                                           \
     blockIdx.z * gridDim.x * gridDim.z) *                              \
    blockDim.x * blockDim.y * blockDim.z)
// Number of blocks in the grid
#define NBG (gridDim.x * gridDim.y * gridDim.z)
// Number of threads in the grid (total number of threads)
#define NTG (blockDim.x * blockDim.y * blockDim.z * \
             gridDim.x * gridDim.y * gridDim.z)
#endif  // __CUDACC__


#endif  // IMGDISTORT_SRC_UTILS_H_
