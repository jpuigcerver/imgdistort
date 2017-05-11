#ifndef IMGDISTORT_SATURATE_CAST_INL_H_
#define IMGDISTORT_SATURATE_CAST_INL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace imgdistort {

/****************************************************************************/
/* SATURATE_CAST TO INT8_T                                                  */
/****************************************************************************/

template<> __host__ __device__
inline int8_t saturate_cast<int8_t>(int16_t v)      {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.s8.s16 %0, %1;" : "=r"(res) : "h"(v));
  return res;
#else
  return (int8_t)((uint32_t)(v-INT8_MIN) <= (uint32_t)UINT8_MAX
      ? v : (v > 0 ? INT8_MAX : INT8_MIN));
#endif
}
template<> __host__ __device__
inline int8_t saturate_cast<int8_t>(int32_t v)      {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.s8.s32 %0, %1;" : "=r"(res) : "r"(v));
  return res;
#else
  return (int8_t)((uint32_t)(v-INT8_MIN) <= (uint32_t)UINT8_MAX
      ? v : (v > 0 ? INT8_MAX : INT8_MIN));
#endif
}
template<> __host__ __device__
inline int8_t saturate_cast<int8_t>(int64_t v)      {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.s8.s64 %0, %1;" : "=r"(res) : "l"(v));
  return res;
#else
  return (int8_t)((uint64_t)((int64_t)v-INT8_MIN) <= (uint64_t)UINT8_MAX
      ? v
      : v > 0 ? INT8_MAX : INT8_MIN);
#endif
}
template<> __host__ __device__
inline int8_t saturate_cast<int8_t>(uint8_t v)      {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  uint32_t vi = v;
  asm("cvt.sat.s8.u8 %0, %1;" : "=r"(res) : "r"(vi));
  return res;
#else
  return static_cast<int8_t>(std::min<uint8_t>(v, INT8_MAX));
#endif
}
template<> __host__ __device__
inline int8_t saturate_cast<int8_t>(uint16_t v)     {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.s8.u16 %0, %1;" : "=r"(res) : "h"(v));
  return res;
#else
  return static_cast<int8_t>(std::min<uint16_t>(v, INT8_MAX));
#endif
}
template<> __host__ __device__
inline int8_t saturate_cast<int8_t>(uint32_t v)     {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.s8.u32 %0, %1;" : "=r"(res) : "r"(v));
  return res;
#else
  return static_cast<int8_t>(std::min<uint32_t>(v, INT8_MAX));
#endif
}
template<> __host__ __device__
inline int8_t saturate_cast<int8_t>(uint64_t v)     {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.s8.u64 %0, %1;" : "=r"(res) : "l"(v));
  return res;
#else
  return static_cast<uint8_t>(std::min<uint64_t>(v, INT8_MAX));
#endif
}
template<> __host__ __device__
inline int8_t saturate_cast<int8_t>(float v)        {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(res) : "f"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<int8_t>(iv);
#endif
}
template<> __host__ __device__
inline int8_t saturate_cast<int8_t>(double v)       {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.rni.sat.u8.f64 %0, %1;" : "=r"(res) : "d"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<int8_t>(iv);
#endif
}


/****************************************************************************/
/* SATURATE_CAST TO INT16_T                                                 */
/****************************************************************************/

template<> __host__ __device__
inline int16_t saturate_cast<int16_t>(int32_t v)    {
#ifdef __CUDA_ARCH__
  int16_t res = 0;
  asm("cvt.sat.s16.s32 %0, %1;" : "=h"(res) : "r"(v));
  return res;
#else
  return static_cast<int16_t>(
      (uint32_t)(v - INT16_MIN) <= (uint32_t)UINT16_MAX
      ? v : v > 0 ? INT16_MAX : INT16_MIN);
#endif
}
template<> __host__ __device__
inline int16_t saturate_cast<int16_t>(int64_t v)    {
#ifdef __CUDA_ARCH__
  int16_t res = 0;
  asm("cvt.sat.s16.s64 %0, %1;" : "=h"(res) : "l"(v));
  return res;
#else
  return static_cast<int16_t>(
      (uint64_t)((int64_t)v - INT16_MIN) <= (uint64_t)UINT16_MAX
      ? v : v > 0 ? INT16_MAX : INT16_MIN);
#endif
}
template<> __host__ __device__
inline int16_t saturate_cast<int16_t>(uint16_t v)   {
#ifdef __CUDA_ARCH__
  int16_t res = 0;
  asm("cvt.sat.s16.u16 %0, %1;" : "=h"(res) : "h"(v));
  return res;
#else
  return static_cast<int16_t>(std::min<int32_t>(v, INT16_MAX));
#endif
}
template<> __host__ __device__
inline int16_t saturate_cast<int16_t>(uint32_t v)   {
#ifdef __CUDA_ARCH__
  int16_t res = 0;
  asm("cvt.sat.s16.u32 %0, %1;" : "=h"(res) : "r"(v));
  return res;
#else
  return static_cast<int16_t>(std::min<uint32_t>(v, INT16_MAX));
#endif
}
template<> __host__ __device__
inline int16_t saturate_cast<int16_t>(uint64_t v)   {
#ifdef __CUDA_ARCH__
  int16_t res = 0;
  asm("cvt.sat.s16.u64 %0, %1;" : "=h"(res) : "l"(v));
  return res;
#else
  return static_cast<int16_t>(std::min<uint64_t>(v, INT16_MAX));
#endif
}
template<> __host__ __device__
inline int16_t saturate_cast<int16_t>(float v)      {
#ifdef __CUDA_ARCH__
  int16_t res = 0;
  asm("cvt.rni.sat.s16.f32 %0, %1;" : "=h"(res) : "f"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<int16_t>(iv);
#endif
}
template<> __host__ __device__
inline int16_t saturate_cast<int16_t>(double v)     {
#ifdef __CUDA_ARCH__
  int16_t res = 0;
  asm("cvt.rni.sat.s16.f64 %0, %1;" : "=h"(res) : "d"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<int16_t>(iv);
#endif
}


/****************************************************************************/
/* SATURATE_CAST TO INT32_T                                                 */
/****************************************************************************/

template<> __host__ __device__
inline int32_t saturate_cast<int32_t>(float v)      {
#ifdef __CUDA_ARCH__
  int32_t res = 0;
  asm("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(res) : "f"(v));
  return res;
#else
  return std::round(v);
#endif
}
template<> __host__ __device__
inline int32_t saturate_cast<int32_t>(double v)     {
#ifdef __CUDA_ARCH__
  int32_t res = 0;
  asm("cvt.rni.sat.s32.f64 %0, %1;" : "=r"(res) : "d"(v));
  return res;
#else
  return std::round(v);
#endif
}


/****************************************************************************/
/* SATURATE_CAST TO INT64_T                                                 */
/****************************************************************************/

template<> __host__ __device__
inline int64_t saturate_cast<int64_t>(float v)      {
#ifdef __CUDA_ARCH__
  int64_t res = 0;
  asm("cvt.rni.sat.s64.f32 %0, %1;" : "=l"(res) : "f"(v));
  return res;
#else
  return std::round(v);
#endif
}
template<> __host__ __device__
inline int64_t saturate_cast<int64_t>(double v)     {
#ifdef __CUDA_ARCH__
  int64_t res = 0;
  asm("cvt.rni.sat.s64.f64 %0, %1;" : "=l"(res) : "d"(v));
  return res;
#else
  return std::round(v);
#endif
}


/****************************************************************************/
/* SATURATE_CAST TO UINT8_T                                                 */
/****************************************************************************/

template<> __host__ __device__
inline uint8_t saturate_cast<uint8_t>(int8_t v)     {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  int32_t vi = v;
  asm("cvt.sat.u8.s8 %0, %1;" : "=r"(res) : "r"(vi));
  return res;
#else
  return static_cast<uint8_t>(std::max<int8_t>(v, 0));
#endif
}
template<> __host__ __device__
inline uint8_t saturate_cast<uint8_t>(int16_t v)    {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.u8.s16 %0, %1;" : "=r"(res) : "h"(v));
  return res;
#else
  return static_cast<uint8_t>((uint32_t)v <= (uint32_t)UINT8_MAX
      ? v : (v > 0 ? UINT8_MAX : 0));
#endif
}
template<> __host__ __device__
inline uint8_t saturate_cast<uint8_t>(int32_t v)    {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.u8.s32 %0, %1;" : "=r"(res) : "r"(v));
  return res;
#else
  return static_cast<uint8_t>((uint32_t)v <= (uint32_t)UINT8_MAX
      ? v : (v > 0 ? UINT8_MAX : 0));
#endif
}
template<> __host__ __device__
inline uint8_t saturate_cast<uint8_t>(int64_t v)    {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.u8.s64 %0, %1;" : "=r"(res) : "l"(v));
  return res;
#else
  return static_cast<uint8_t>((uint64_t)v <= (uint64_t)UINT8_MAX
      ? v : (v > 0 ? UINT8_MAX : 0));
#endif
}
template<> __host__ __device__
inline uint8_t saturate_cast<uint8_t>(uint16_t v)   {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.u8.u16 %0, %1;" : "=r"(res) : "h"(v));
  return res;
#else
  return static_cast<uint8_t>(std::min<uint16_t>(v, UINT8_MAX));
#endif
}
template<> __host__ __device__
inline uint8_t saturate_cast<uint8_t>(uint32_t v)   {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.u8.u32 %0, %1;" : "=r"(res) : "r"(v));
  return res;
#else
  return static_cast<uint8_t>(std::min<uint32_t>(v, UINT8_MAX));
#endif
}
template<> __host__ __device__
inline uint8_t saturate_cast<uint8_t>(uint64_t v)   {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.u8.u64 %0, %1;" : "=r"(res) : "d"(v));
  return res;
#else
  return static_cast<uint8_t>(std::min<uint64_t>(v, UINT8_MAX));
#endif
}
template<> __host__ __device__
inline uint8_t saturate_cast<uint8_t>(float v)      {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(res) : "f"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<uint8_t>(iv);
#endif
}
template<> __host__ __device__
inline uint8_t saturate_cast<uint8_t>(double v)     {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.rni.sat.u8.f64 %0, %1;" : "=r"(res) : "d"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<uint8_t>(iv);
#endif
}


/****************************************************************************/
/* SATURATE_CAST TO UINT16_T                                                */
/****************************************************************************/

template<> __host__ __device__
inline uint16_t saturate_cast<uint16_t>(int8_t v)   {
#ifdef __CUDA_ARCH__
  uint16_t res = 0;
  int32_t vi = v;
  asm("cvt.sat.u16.s8 %0, %1;" : "=h"(res) : "r"(vi));
  return res;
#else
  return static_cast<uint16_t>(std::max<int8_t>(v, 0));
#endif
}
template<> __host__ __device__
inline uint16_t saturate_cast<uint16_t>(int16_t v)  {
#ifdef __CUDA_ARCH__
  uint16_t res = 0;
  asm("cvt.sat.u16.s16 %0, %1;" : "=h"(res) : "h"(v));
  return res;
#else
  return static_cast<uint16_t>(std::max<int16_t>(v, 0));
#endif
}
template<> __host__ __device__
inline uint16_t saturate_cast<uint16_t>(int32_t v)  {
#ifdef __CUDA_ARCH__
  uint16_t res = 0;
  asm("cvt.sat.u16.s32 %0, %1;" : "=h"(res) : "r"(v));
  return res;
#else
  return static_cast<uint16_t>((uint32_t)v <= (uint32_t)UINT16_MAX
      ? v : (v > 0 ? UINT16_MAX : 0));
#endif
}
template<> __host__ __device__
inline uint16_t saturate_cast<uint16_t>(int64_t v)  {
#ifdef __CUDA_ARCH__
  uint16_t res = 0;
  asm("cvt.sat.u16.s64 %0, %1;" : "=h"(res) : "l"(v));
  return res;
#else
  return static_cast<uint16_t>((uint64_t)v <= (uint64_t)UINT16_MAX
      ? v : (v > 0 ? UINT16_MAX : 0));
#endif
}
template<> __host__ __device__
inline uint16_t saturate_cast<uint16_t>(uint32_t v) {
#ifdef __CUDA_ARCH__
  uint16_t res = 0;
  asm("cvt.sat.u16.u32 %0, %1;" : "=h"(res) : "r"(v));
  return res;
#else
  return static_cast<uint16_t>(std::min<uint32_t>(v, UINT16_MAX));
#endif
}
template<> __host__ __device__
inline uint16_t saturate_cast<uint16_t>(uint64_t v) {
#ifdef __CUDA_ARCH__
  uint16_t res = 0;
  asm("cvt.sat.u16.u64 %0, %1;" : "=h"(res) : "l"(v));
  return res;
#else
  return static_cast<uint16_t>(std::min<uint64_t>(v, UINT16_MAX));
#endif
}
template<> __host__ __device__
inline uint16_t saturate_cast<uint16_t>(float v)    {
#ifdef __CUDA_ARCH__
  uint16_t res = 0;
  asm("cvt.rni.sat.u16.f32 %0, %1;" : "=h"(res) : "f"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<uint16_t>(iv);
#endif
}
template<> __host__ __device__
inline uint16_t saturate_cast<uint16_t>(double v)   {
#ifdef __CUDA_ARCH__
  uint16_t res = 0;
  asm("cvt.rni.sat.u16.f64 %0, %1;" : "=h"(res) : "d"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<uint16_t>(iv);
#endif
}


/****************************************************************************/
/* SATURATE_CAST TO UINT32_T                                                */
/****************************************************************************/

template<> __host__ __device__
inline uint32_t saturate_cast<uint32_t>(int8_t v) {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  int32_t vi = v;
  asm("cvt.sat.u32.s8 %0, %1;" : "=r"(res) : "r"(vi));
  return res;
#else
  return (uint32_t)std::max<int8_t>(v, 0);
#endif
}
template<> __host__ __device__
inline uint32_t saturate_cast<uint32_t>(int16_t v) {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.u32.s16 %0, %1;" : "=r"(res) : "h"(v));
  return res;
#else
  return static_cast<uint32_t>(std::max<int16_t>(v, 0));
#endif
}
template<> __host__ __device__
inline uint32_t saturate_cast<uint32_t>(int32_t v) {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.u32.s32 %0, %1;" : "=r"(res) : "r"(v));
  return res;
#else
  return static_cast<uint32_t>(std::max<int32_t>(v, 0));
#endif
}
template<> __host__ __device__
inline uint32_t saturate_cast<uint32_t>(int64_t v) {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.sat.u32.s64 %0, %1;" : "=r"(res) : "l"(v));
  return res;
#else
  return static_cast<uint32_t>((uint64_t)v <= (uint64_t)UINT32_MAX
      ? v : (v > 0 ? UINT32_MAX : 0));
#endif
}
template<> __host__ __device__
inline uint32_t saturate_cast<uint32_t>(float v)    {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.rni.sat.u32.f32 %0, %1;" : "=r"(res) : "f"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<uint32_t>(iv);
#endif
}
template<> __host__ __device__
inline uint32_t saturate_cast<uint32_t>(double v)   {
#ifdef __CUDA_ARCH__
  uint32_t res = 0;
  asm("cvt.rni.sat.u32.f64 %0, %1;" : "=r"(res) : "d"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<uint32_t>(iv);
#endif
}


/****************************************************************************/
/* SATURATE_CAST TO UINT64_T                                                */
/****************************************************************************/

template<> __host__ __device__
inline uint64_t saturate_cast<uint64_t>(int8_t v) {
#ifdef __CUDA_ARCH__
  uint64_t res = 0;
  int32_t vi = v;
  asm("cvt.sat.u64.s8 %0, %1;" : "=l"(res) : "r"(vi));
  return res;
#else
  return static_cast<uint64_t>(std::max<int8_t>(v, 0));
#endif
}
template<> __host__ __device__
inline uint64_t saturate_cast<uint64_t>(int16_t v) {
#ifdef __CUDA_ARCH__
  uint64_t res = 0;
  asm("cvt.sat.u64.s16 %0, %1;" : "=l"(res) : "h"(v));
  return res;
#else
  return static_cast<uint64_t>(std::max<int16_t>(v, 0));
#endif
}
template<> __host__ __device__
inline uint64_t saturate_cast<uint64_t>(int32_t v) {
#ifdef __CUDA_ARCH__
  uint64_t res = 0;
  asm("cvt.sat.u64.s32 %0, %1;" : "=l"(res) : "r"(v));
  return res;
#else
  return static_cast<uint64_t>(std::max<int32_t>(v, 0));
#endif
}
template<> __host__ __device__
inline uint64_t saturate_cast<uint64_t>(int64_t v) {
#ifdef __CUDA_ARCH__
  uint64_t res = 0;
  asm("cvt.sat.u64.s64 %0, %1;" : "=l"(res) : "l"(v));
  return res;
#else
  return static_cast<uint64_t>(std::max<int64_t>(v, 0));
#endif
}
template<> __host__ __device__
inline uint64_t saturate_cast<uint64_t>(float v)    {
#ifdef __CUDA_ARCH__
  uint64_t res = 0;
  asm("cvt.rni.sat.u64.f32 %0, %1;" : "=l"(res) : "f"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<uint64_t>(iv);
#endif
}
template<> __host__ __device__
inline uint64_t saturate_cast<uint64_t>(double v)   {
#ifdef __CUDA_ARCH__
  uint64_t res = 0;
  asm("cvt.rni.sat.u64.f64 %0, %1;" : "=l"(res) : "d"(v));
  return res;
#else
  const int iv = std::round(v); return saturate_cast<uint64_t>(iv);
#endif
}

}  // namespace imgdistort

#endif  // IMGDISTORT_SATURATE_CAST_INL_H_
