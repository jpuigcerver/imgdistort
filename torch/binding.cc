#include <TH.h>
#include <THTensor.h>

#if defined(WITH_CUTORCH) && defined(WITH_CUDA)
#include <THC.h>
#include <THCTensor.h>
#endif

#include <lua.hpp>
#include <luaT.h>

#include <string>

#include "utils.h"

#include "THTemplates.h"

#include "WrapAffine.h"

namespace imgdistort {
namespace torch {

template <typename AffineTensorType, typename TensorType, typename DataType,
          typename StateType, typename StreamType>
int affine_nchw(lua_State* L,
                const std::string& affine_ttype_str,
                const std::string& ttype_str) {
  // Get affine, input and output tensors.
  const AffineTensorType* M = static_cast<AffineTensorType*>(
      luaT_checkudata(L, 1, affine_ttype_str.c_str()));
  TensorType* src = static_cast<TensorType*>(
      luaT_toudata(L, 2, ttype_str.c_str()));
  TensorType* dst = static_cast<TensorType*>(
      luaT_toudata(L, 3, ttype_str.c_str()));

  // Get Torch state (only used for GPUs, actually)
  StateType* state = THGetState<StateType>(L);

  // Check affine transformation matrix.
  if (M->nDimension != 3 ||
      M->size[0] == 0 || M->size[1] != 2 || M->size[2] != 3) {
    lua_pushfstring(L, "Affine tensor must have a ?x2x3 shape.");
    lua_error(L);
  }
  if (THTensor_isContiguous(M)) {
    lua_pushfstring(L, "Affine tensor must be contiguous.");
    lua_error(L);
  }
  // Check input tensor.
  if (THTensor_nElement(src) == 0 || src->nDimension != 4) {
    lua_pushfstring(
        L, "Input tensor must have 4 dimensions and cannot be empty.");
    lua_error(L);
  }
  if (THTensor_isContiguous(src)) {
    lua_pushfstring(L, "Input tensor must be contiguous.");
    lua_error(L);
  }
  // Resize output tensor to have the same dimensions as the input
  THTensor_resizeAs(dst, src, state);

  const int Mn = M->size[0];
  const int N = src->size[0];
  const int C = src->size[1];
  const int H = src->size[2];
  const int W = src->size[3];

  const double* M_ptr =
      static_cast<const double*>(M->storage->data + M->storageOffset);
  const DataType* src_ptr =
      static_cast<const DataType*>(src->storage->data + src->storageOffset);
  DataType* dst_ptr =
      static_cast<DataType*>(dst->storage->data + dst->storageOffset);

  wrap_affine_call<TensorType, DataType>(
      N, C, H, W, M_ptr, Mn, src_ptr, dst_ptr, THGetStream<StreamType>(L));
  return 0;
}

template <typename TT, typename DT>
int affine_nchw_cpu(lua_State* L, const std::string& ttype) {
  return ::imgdistort::torch::affine_nchw<
    THDoubleTensor, TT, DT, void, void*>(
        L, "torch.DoubleTensor", ttype);
}

template <typename TT, typename DT>
int affine_nchw_gpu(lua_State* L, const std::string& ttype) {
  return ::imgdistort::torch::affine_nchw<
    THCudaDoubleTensor, TT, DT, THCState, cudaStream_t>(
        L, "torch.CudaDoubleTensor", ttype);
}

}  // namespace torch
}  // namespace imgdistort

TORCH_API int imgdistort_affine_nchw(lua_State* L) {
  const std::string ttype = luaT_typename(L, 2);
  const std::string dst_type = luaT_typename(L, 3);
  if (ttype != std::string(luaT_typename(L, 3))) {
    lua_pushfstring(
        L, "Input and output tensors for imgdistort_affine_nchw must have "
        "the same type.");
    lua_error(L);
  }

  // CPU, float
  if (ttype == "torch.FloatTensor") {
    return ::imgdistort::torch::affine_nchw_cpu<THFloatTensor, float>(
        L, ttype);
  }
  // CPU, double
  if (ttype == "torch.DoubleTensor") {
    return ::imgdistort::torch::affine_nchw_cpu<THDoubleTensor, double>(
        L, ttype);
  }
  // CPU, uint8_t
  if (ttype == "torch.ByteTensor") {
    return ::imgdistort::torch::affine_nchw_cpu<THByteTensor, uint8_t>(
        L, ttype);
  }
  // CPU, int16_t
  if (ttype == "torch.ShortTensor") {
    return ::imgdistort::torch::affine_nchw_cpu<THShortTensor, int16_t>(
        L, ttype);
  }
  // CPU, int32_t
  if (ttype == "torch.IntTensor") {
    return ::imgdistort::torch::affine_nchw_cpu<THIntTensor, int32_t>(
        L, ttype);
  }
  // CPU, int64_t
  if (ttype == "torch.LongTensor") {
    return ::imgdistort::torch::affine_nchw_cpu<THLongTensor, int64_t>(
        L, ttype);
  }

#if defined(WITH_CUDA) && defined(WITH_CUTORCH)
  // GPU, float
  if (ttype == "torch.CudaTensor") {
    return ::imgdistort::torch::affine_nchw_gpu<THCudaTensor, float>(
        L, ttype);
  }
   // GPU, double
  if (ttype == "torch.CudaDoubleTensor") {
    return ::imgdistort::torch::affine_nchw_gpu<THCudaDoubleTensor, double>(
        L, ttype);
  }
  // GPU, uint8_t
  if (ttype == "torch.CudaByteTensor") {
    return ::imgdistort::torch::affine_nchw_gpu<THCudaByteTensor, uint8_t>(
        L, ttype);
  }
  // GPU, int16_t
  if (ttype == "torch.CudaShortTensor") {
    return ::imgdistort::torch::affine_nchw_gpu<THCudaShortTensor, int16_t>(
        L, ttype);
  }
  // GPU, int32_t
  if (ttype == "torch.CudaIntTensor") {
    return ::imgdistort::torch::affine_nchw_gpu<THCudaIntTensor, int32_t>(
        L, ttype);
  }
  // GPU, int64_t
  if (ttype == "torch.CudaLongTensor") {
    return ::imgdistort::torch::affine_nchw_gpu<THCudaLongTensor, int64_t>(
        L, ttype);
  }
#endif

  const std::string error =
      "imgdistort_affine_nchw is not implemented for tensors of type " + ttype;
  lua_pushfstring(L, error.c_str());
  lua_error(L);
  return -1;
}

/*
template <bool dilate>
int dilate_or_erode_NCHW(lua_State* L) {
  THCState* state = cutorch_getstate(L);
  THCudaTensor* inp = static_cast<THCudaTensor *>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* out = static_cast<THCudaTensor *>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaByteTensor* M = static_cast<THCudaByteTensor *>(
      luaT_checkudata(L, 3, "torch.CudaByteTensor"));
  // Check input data
  if (inp->storage == 0) {
    lua_pushfstring(L, "input cannot be an empty tensor");
    lua_error(L);
  }
  if (inp->nDimension != 4 || THCudaTensor_nElement(state, inp) < 1) {
    lua_pushfstring(L, "input must have 4 dimensions and be non-empty");
    lua_error(L);
  }
  if (!THCudaTensor_isContiguous(state, inp)) {
    lua_pushfstring(L, "input tensor must be continuous");
    lua_error(L);
  }
  // Check structure kernel size
  if (M->nDimension != 3 || THCudaByteTensor_nElement(state, M) < 1) {
    lua_pushfstring(L,
                    "structure tensor must have 3 dimensions and be non-empty");
    lua_error(L);
  }
  if (!THCudaByteTensor_isContiguous(state, M)) {
    lua_pushfstring(L, "structure tensor must be continuous");
    lua_error(L);
  }
  // Resize output tensor to have the same dimensions as the input
  THCudaTensor_resizeAs(state, out, inp);

  cudaStream_t stream = THCState_getCurrentStream(state);
  const float* inp_data = THCudaTensor_data(state, inp);
  float* out_data = THCudaTensor_data(state, out);
  const bool* M_data =
      reinterpret_cast<bool*>(THCudaByteTensor_data(state, M));

  if (dilate) {
    dilate_NCHW_f32(out_data, inp_data,
                    inp->size[0], inp->size[1], inp->size[2], inp->size[3],
                    M_data, M->size[0], M->size[1], M->size[2],
                    stream);
  } else {
    erode_NCHW_f32(out_data, inp_data,
                   inp->size[0], inp->size[1], inp->size[2], inp->size[3],
                   M_data, M->size[0], M->size[1], M->size[2],
                   stream);
  }

  return 0;
}

TORCH_API int dilate_NCHW(lua_State* L) {
  return dilate_or_erode_NCHW<true>(L);
}

TORCH_API int erode_NCHW(lua_State* L) {
  return dilate_or_erode_NCHW<false>(L);
}

*/
TORCH_API int luaopen_libimgdistort(lua_State *L) {
  lua_register(L, "imgdistort_affine_nchw", imgdistort_affine_nchw);
  //lua_register(L, "dilate_NCHW", dilate_NCHW);
  //lua_register(L, "erode_NCHW",  erode_NCHW);
  return 0;
}
