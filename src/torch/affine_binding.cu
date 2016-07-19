#include <lua.hpp>
#include <luaT.h>
#include <THC.h>
#include <THCTensor.h>

#include "affine.cuh"
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
  if (inp->nDimension != 4) {
    lua_pushfstring(L, "input must have 4 dimensions");
    lua_error(L);
  }
  if (inp->size[0] <= 0 || inp->size[1] <= 0 || inp->size[2] <= 0 ||
      inp->size[3] <= 0) {
    lua_pushfstring(
        L, "all dimensions of the input must be greater than or equal to 1");
    lua_error(L);
  }
  if (!THCudaTensor_isContiguous(state, inp)) {
    lua_pushfstring(L, "input tensor must be continuous");
    lua_error(L);
  }
  // Check transformation matrix
  if (M->nDimension != 3 || M->size[0] != inp->size[0] ||
      M->size[1] != 2 || M->size[2] != 3) {
    lua_pushfstring(L, "transformation tensor must be a nx2x3 tensor");
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

  affine_NCHW(inp->size[0], inp->size[1], inp->size[2], inp->size[3],
              out_data, inp_data, M_data, stream);

  return 0;
}

TORCH_API int luaopen_libimgdistort(lua_State *L) {
  lua_register(L, "affine_NCHW", affine_NCHW);
  return 0;
}
