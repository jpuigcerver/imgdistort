#ifndef IMGDISTORT_TORCH_THTEMPLATES_H_
#define IMGDISTORT_TORCH_THTEMPLATES_H_

#include <luaT.h>
#include <lua.hpp>

#include <TH.h>
#include <THTensor.h>
#if defined(WITH_CUTORCH) && defined(WITH_CUDA)
#include <THC.h>
#include <THCTensor.h>
#endif

#include "utils.h"

#define DEFINE_CPU_IMPLEMENTATION(TensorType)                           \
  template <>                                                           \
  void THTensor_resizeAs<TensorType>(TensorType* t,  TensorType* r,     \
                                     void* s) {                         \
    TensorType ## _resizeAs(t, r);                                      \
  }                                                                     \
                                                                        \
  template                                                              \
  void THTensor_resizeAs<TensorType>(TensorType*, TensorType*, void*)

#define DEFINE_GPU_IMPLEMENTATION(TensorType)                           \
  template <>                                                           \
  void THTensor_resizeAs<TensorType>(TensorType* t,  TensorType* r,     \
                                     THCState* s) {                     \
    TensorType ## _resizeAs(s, t, r);                                   \
  }                                                                     \
                                                                        \
  template                                                              \
  void THTensor_resizeAs<TensorType>(TensorType*, TensorType*, THCState*)

namespace imgdistort {
namespace torch {

template <typename State>
inline State* THGetState(lua_State* L) {
  return nullptr;
}

template <typename Stream>
inline Stream THGetStream(lua_State* L) {
  return nullptr;
}

template <typename Tensor>
long THTensor_nElement(const Tensor* t) {
  if (t->nDimension == 0) return 0;
  long nElement = 1;
  for (int d = 0; d < t->nDimension; ++d) { nElement *= t->size[d]; }
  return nElement;
}

template <typename Tensor>
bool THTensor_isContiguous(const Tensor* t) {
  long z = 1;
  int d;
  for(d = t->nDimension-1; d >= 0; d--) {
    if(t->size[d] != 1) {
      if(t->stride[d] == z)
        z *= t->size[d];
      else
        return false;
    }
  }
  return true;
}

template <typename Tensor, typename State>
void THTensor_resizeAs(Tensor* t, Tensor* r, State* s);

DEFINE_CPU_IMPLEMENTATION(THFloatTensor);
DEFINE_CPU_IMPLEMENTATION(THDoubleTensor);
DEFINE_CPU_IMPLEMENTATION(THByteTensor);
DEFINE_CPU_IMPLEMENTATION(THShortTensor);
DEFINE_CPU_IMPLEMENTATION(THIntTensor);
DEFINE_CPU_IMPLEMENTATION(THLongTensor);

// GPU IMPLEMENTATIONS
#if defined(WITH_CUDA) && defined(WITH_CUTORCH)
template <>
inline THCState* THGetState(lua_State* L) {
  return cutorch_getstate(L);
}

template <>
inline cudaStream_t THGetStream(lua_State* L) {
  THCState* state = cutorch_getstate(L);
  return THCState_getCurrentStream(state);
}

DEFINE_GPU_IMPLEMENTATION(THCudaTensor);
DEFINE_GPU_IMPLEMENTATION(THCudaDoubleTensor);
DEFINE_GPU_IMPLEMENTATION(THCudaByteTensor);
DEFINE_GPU_IMPLEMENTATION(THCudaShortTensor);
DEFINE_GPU_IMPLEMENTATION(THCudaIntTensor);
DEFINE_GPU_IMPLEMENTATION(THCudaLongTensor);
#endif

}  // namespace torch
}  // namespace imgdistort

#endif  // IMGDISTORT_TORCH_THTEMPLATES_H_
