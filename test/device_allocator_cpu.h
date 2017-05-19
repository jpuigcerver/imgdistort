#ifndef IMGDISTORT_DEVICE_ALLOCATOR_CPU_H_
#define IMGDISTORT_DEVICE_ALLOCATOR_CPU_H_

#include "device_allocator.h"

#include <cstring>
#include <random>
#include <vector>

namespace imgdistort {
namespace testing {

template <>
template <typename T>
T* DeviceAllocator<CPU>::Allocate(size_t n) {
  return new T[n];
}

template <>
template <typename T>
void DeviceAllocator<CPU>::Deallocate(T* ptr) {
  delete [] ptr;
}

template <>
template <typename T>
void DeviceAllocator<CPU>::CopyToHost(size_t n, const T* src, T* dst) {
  memcpy(dst, src, n * sizeof(T));
}

template <>
template <typename T>
void DeviceAllocator<CPU>::CopyToDevice(size_t n, const T* src, T* dst) {
  memcpy(dst, src, n * sizeof(T));
}

template <>
template <typename T>
T* DeviceAllocator<CPU>::GenerateRandom(size_t n) {
  std::default_random_engine rng;
  std::uniform_int_distribution<T> dist(-1, 1);
  T* data = new T[n];
  for (size_t i = 0; i < n; ++i) { data[i] = dist(rng); }
  return data;
}

template <>
template <>
float* DeviceAllocator<CPU>::GenerateRandom<float>(size_t n) {
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  float* data = new float[n];
  for (size_t i = 0; i < n; ++i) { data[i] = dist(rng); }
  return data;
}

template <>
template <>
double* DeviceAllocator<CPU>::GenerateRandom<double>(size_t n) {
  std::default_random_engine rng;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  double* data = new double[n];
  for (size_t i = 0; i < n; ++i) { data[i] = dist(rng); }
  return data;
}

}  // namespace testing
}  // namespace imgdistort

#endif // IMGDISTORT_DEVICE_ALLOCATOR_CPU_H_
