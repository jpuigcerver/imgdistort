#ifndef IMGDISTORT_GPU_DEFINES_H_
#define IMGDISTORT_GPU_DEFINES_H_

#define DIV_UP(x, y) ((x) == 0 ? 0 : 1 + ((x) - 1) / (y))

#define NUM_BLOCKS(n, s) std::min<int>(DIV_UP(n, s), 65535)

// Thread IDs within a block
#define thBx (threadIdx.x)
#define thBy (threadIdx.y)
#define thBz (threadIdx.z)
#define thBi (                                          \
    threadIdx.x +                                       \
    threadIdx.y * blockDim.x +                          \
    threadIdx.z * blockDim.x * blockDim.y)

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

// Number of threads within the grid, in each dimension
#define NTGx (blockDim.x * gridDim.x)
#define NTGy (blockDim.y * gridDim.y)
#define NTGz (blockDim.z * gridDim.z)

// Number of threads in a block
#define NTB (blockDim.x * blockDim.y * blockDim.z)
// Number of blocks in the grid
#define NBG (gridDim.x * gridDim.y * gridDim.z)
// Number of threads in the grid (total number of threads)
#define NTG (blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z)

#endif  // IMGDISTORT_GPU_DEFINES_H_