#ifndef IMGDISTORT_DEVICE_ALLOCATOR_GPU_H_
#define IMGDISTORT_DEVICE_ALLOCATOR_GPU_H_

#include "device_allocator.h"

#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <imgdistort/defines.h>
#include <glog/logging.h>

namespace imgdistort {
namespace testing {

template <>
template <typename T>
T* DeviceAllocator<GPU>::Allocate(size_t n) {
  T* ptr = nullptr;
  CHECK_CUDA_CALL(cudaMalloc(&ptr, sizeof(T) * n));
  return ptr;
}

template <>
template <typename T>
void DeviceAllocator<GPU>::Deallocate(T* ptr) {
  CHECK_CUDA_CALL(cudaFree((void*)ptr));
}

template <>
template <typename T>
void DeviceAllocator<GPU>::CopyToHost(size_t n, const T* src, T* dst) {
  CHECK_CUDA_CALL(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToHost));
}

template <>
template <typename T>
void DeviceAllocator<GPU>::CopyToDevice(size_t n, const T* src, T* dst) {
  CHECK_CUDA_CALL(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyHostToDevice));
}

template <>
template <typename T>
T* DeviceAllocator<GPU>::GenerateRandom(size_t n) {
  std::default_random_engine rng;
  std::uniform_int_distribution<T> dist(-1, 1);
  std::vector<T> tmp(n);
  for (size_t i = 0; i < n; ++i) { tmp[i] = dist(rng); }
  const size_t nbytes = n * sizeof(T);
  T* data = nullptr;
  CHECK_CUDA_CALL(cudaMalloc(&data, nbytes));
  CHECK_CUDA_CALL(cudaMemcpy(data, tmp.data(), nbytes, cudaMemcpyHostToDevice));
  return data;
}

template <>
template <>
float* DeviceAllocator<GPU>::GenerateRandom<float>(size_t n) {
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  float* tmp = new float[n];
  for (size_t i = 0; i < n; ++i) { tmp[i] = dist(rng); }
  const size_t nbytes = n * sizeof(float);
  float* data = nullptr;
  CHECK_CUDA_CALL(cudaMalloc(&data, nbytes));
  CHECK_CUDA_CALL(cudaMemcpy(data, tmp, nbytes, cudaMemcpyHostToDevice));
  return data;
}

template <>
template <>
double* DeviceAllocator<GPU>::GenerateRandom<double>(size_t n) {
  std::default_random_engine rng;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  double* tmp = new double[n];
  for (size_t i = 0; i < n; ++i) { tmp[i] = dist(rng); }
  const size_t nbytes = n * sizeof(double);
  double* data = nullptr;
  CHECK_CUDA_CALL(cudaMalloc(&data, nbytes));
  CHECK_CUDA_CALL(cudaMemcpy(data, tmp, nbytes, cudaMemcpyHostToDevice));
  return data;
}

}  // namespace testing
}  // namespace imgdistort

#endif // IMGDISTORT_DEVICE_ALLOCATOR_GPU_H_
