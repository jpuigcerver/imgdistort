#include <lua.hpp>
#include <luaT.h>
#include <THC.h>
#include <THCTensor.h>

#include "affine.h"
#include "morphology.h"
#include "utils.h"

TORCH_API int affine_NCHW(lua_State* L) {
  THCState* state = cutorch_getstate(L);
  THCudaTensor* inp = static_cast<THCudaTensor *>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* out = static_cast<THCudaTensor *>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* M = static_cast<THCudaTensor *>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
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
  // Check transformation matrix
  if (M->nDimension != 3 || M->size[1] != 2 || M->size[2] != 3) {
    lua_pushfstring(L, "transformation tensor must be a ?x2x3 tensor");
    lua_error(L);
  }
  if (!THCudaTensor_isContiguous(state, M)) {
    lua_pushfstring(L, "transformation tensor must be continuous");
    lua_error(L);
  }
  // Resize output tensor to have the same dimensions as the input
  THCudaTensor_resizeAs(state, out, inp);

  cudaStream_t stream = THCState_getCurrentStream(state);
  const float* inp_data = THCudaTensor_data(state, inp);
  float* out_data = THCudaTensor_data(state, out);
  const float* M_data = THCudaTensor_data(state, M);

  affine_NCHW_f32(out_data, inp_data,
                  inp->size[0], inp->size[1], inp->size[2], inp->size[3],
                  M_data, M->size[0], stream);

  return 0;
}

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

TORCH_API int luaopen_libimgdistort(lua_State *L) {
  lua_register(L, "affine_NCHW", affine_NCHW);
  lua_register(L, "dilate_NCHW", dilate_NCHW);
  lua_register(L, "erode_NCHW",  erode_NCHW);
  return 0;
}
