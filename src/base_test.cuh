#ifndef CUDA_KERNELS_BASE_TEST_CUH_
#define CUDA_KERNELS_BASE_TEST_CUH_

#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "utils.cuh"

class BaseTest : public ::testing::Test {
 public:
  static const int N = 2, C = 3, H = 4, W = 2;

  BaseTest() : x_cpu_(N * C * H * W), y_cpu_(N * C * H * W) {
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

  virtual ~BaseTest() {
    CHECK_CUDA_CALL(cudaFree(x_gpu_));
    CHECK_CUDA_CALL(cudaFree(y_gpu_));
  }

  virtual void CopyToGPU() {
    CHECK_CUDA_CALL(cudaMemcpy(x_gpu_, x_cpu_.data(), x_size(),
                                cudaMemcpyHostToDevice));
  }

  virtual void CopyToCPU() {
    CHECK_CUDA_CALL(cudaMemcpy(y_cpu_.data(), y_gpu_, y_size(),
                                 cudaMemcpyDeviceToHost));
  }

  int x_size() const { return N * C * H * W * sizeof(float); }
  int y_size() const { return N * C * H * W * sizeof(float); }

  const float* x_cpu() const { return x_cpu_.data(); }
  float* y_cpu() { return y_cpu_.data(); }
  const float* x_gpu() const { return x_gpu_; }
  float* y_gpu() { return y_gpu_; }

 protected:
  std::vector<float> x_cpu_, y_cpu_;
  float *x_gpu_, *y_gpu_;
};

#endif  // CUDA_KERNELS_BASE_TEST_CUH_
