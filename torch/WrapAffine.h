#ifndef IMGDISTORT_TORCH_WRAPAFFINE_H_
#define IMGDISTORT_TORCH_WRAPAFFINE_H_

#include <imgdistort/affine_cpu.h>
#if defined(WITH_CUDA) && defined(WITH_CUTORCH)
#include <imgdistort/affine_gpu.h>
#endif

#include <TH.h>
#include <THTensor.h>
#if defined(WITH_CUDA) && defined(WITH_CUTORCH)
#include <THC.h>
#include <THCTensor.h>
#endif

namespace imgdistort {
namespace torch {

template <typename TensorType, typename DataType, typename StreamType>
void wrap_affine_call(const int N, const int C, const int H, const int W,
                      const double* M, const int Mn,
                      const DataType* src, DataType* dst,
                      StreamType stream);


//
// CPU
//
template <>
void wrap_affine_call<THFloatTensor, float, void*>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const float* src, float* dst, void* stream) {
  ::imgdistort::cpu::affine_nchw<float>(N, C, H, W, M, Mn, src, 0, dst, 0);
}

template <>
void wrap_affine_call<THDoubleTensor, double, void*>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const double* src, double* dst, void* stream) {
  ::imgdistort::cpu::affine_nchw<double>(N, C, H, W, M, Mn, src, 0, dst, 0);
}

template <>
void wrap_affine_call<THByteTensor, uint8_t, void*>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint8_t* src, uint8_t* dst, void* stream) {
  ::imgdistort::cpu::affine_nchw<uint8_t>(N, C, H, W, M, Mn, src, 0, dst, 0);
}

template <>
void wrap_affine_call<THShortTensor, int16_t, void*>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int16_t* src, int16_t* dst, void* stream) {
  ::imgdistort::cpu::affine_nchw<int16_t>(N, C, H, W, M, Mn, src, 0, dst, 0);
}

template <>
void wrap_affine_call<THIntTensor, int32_t, void*>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int32_t* src, int32_t* dst, void* stream) {
  ::imgdistort::cpu::affine_nchw<int32_t>(N, C, H, W, M, Mn, src, 0, dst, 0);
}

template <>
void wrap_affine_call<THLongTensor, int64_t, void*>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int64_t* src, int64_t* dst, void* stream) {
  ::imgdistort::cpu::affine_nchw<int64_t>(N, C, H, W, M, Mn, src, 0, dst, 0);
}


//
// GPU
//
#if defined(WITH_CUDA) && defined(WITH_CUTORCH)
template <>
void wrap_affine_call<THCudaTensor, float, cudaStream_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const float* src, float* dst, cudaStream_t stream) {
  ::imgdistort::gpu::affine_nchw<float>(N, C, H, W, M, Mn, src, 0, dst, 0,
                                        stream);
}

template <>
void wrap_affine_call<THCudaDoubleTensor, double, cudaStream_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const double* src, double* dst, cudaStream_t stream) {
  ::imgdistort::gpu::affine_nchw<double>(N, C, H, W, M, Mn, src, 0, dst, 0,
                                         stream);
}

template <>
void wrap_affine_call<THCudaByteTensor, uint8_t, cudaStream_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const uint8_t* src, uint8_t* dst, cudaStream_t stream) {
  ::imgdistort::gpu::affine_nchw<uint8_t>(N, C, H, W, M, Mn, src, 0, dst, 0,
                                          stream);
}

template <>
void wrap_affine_call<THCudaShortTensor, int16_t, cudaStream_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int16_t* src, int16_t* dst, cudaStream_t stream) {
  ::imgdistort::gpu::affine_nchw<int16_t>(N, C, H, W, M, Mn, src, 0, dst, 0,
                                          stream);
}

template <>
void wrap_affine_call<THCudaIntTensor, int32_t, cudaStream_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int32_t* src, int32_t* dst, cudaStream_t stream) {
  ::imgdistort::gpu::affine_nchw<int32_t>(N, C, H, W, M, Mn, src, 0, dst, 0,
                                          stream);
}

template <>
void wrap_affine_call<THCudaLongTensor, int64_t, cudaStream_t>(
    const int N, const int C, const int H, const int W,
    const double* M, const int Mn,
    const int64_t* src, int64_t* dst, cudaStream_t stream) {
  ::imgdistort::gpu::affine_nchw<int64_t>(N, C, H, W, M, Mn, src, 0, dst, 0,
                                          stream);
}
#endif

}  // namespace torch
}  // namespace imgdistort

#endif  // IMGDISTORT_TORCH_WRAPAFFINE_H_
