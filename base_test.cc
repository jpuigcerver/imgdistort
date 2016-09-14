#include "base_test.h"

#include <cuda_runtime.h>

#include "utils.h"

BaseTest::BaseTest() : x_cpu_(N * C * H * W), y_cpu_(N * C * H * W) {
  CHECK_CUDA_CALL(cudaMalloc(&x_gpu_, sizeof(float) * N * C * H * W));
  CHECK_CUDA_CALL(cudaMalloc(&y_gpu_, sizeof(float) * N * C * H * W));
  // Initialize x in the cpu
  for (int n = 0, i = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x, ++i) {
          x_cpu_[i] = i + 1;
        }
      }
    }
  }
}

BaseTest::~BaseTest() {
  CHECK_CUDA_CALL(cudaFree(x_gpu_));
  CHECK_CUDA_CALL(cudaFree(y_gpu_));
}

void BaseTest::CopyToCPU() {
  CHECK_CUDA_CALL(cudaMemcpy(y_cpu_.data(), y_gpu_, y_size(),
                             cudaMemcpyDeviceToHost));
}

void BaseTest::CopyToGPU() {
  CHECK_CUDA_CALL(cudaMemcpy(x_gpu_, x_cpu_.data(), x_size(),
                             cudaMemcpyHostToDevice));
}
