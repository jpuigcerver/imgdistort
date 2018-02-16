#ifndef IMGDISTORT_SATURATE_CAST_H_
#define IMGDISTORT_SATURATE_CAST_H_

#include <cmath>
#include <cstdint>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace imgdistort {

template <typename T>
__host__ __device__
inline T saturate_cast(double v) { return static_cast<T>(v); }

template <typename T>
__host__ __device__
inline T saturate_cast(float v) { return static_cast<T>(v); }

template <typename T>
__host__ __device__
inline T saturate_cast(int8_t v) { return static_cast<T>(v); }

template <typename T>
__host__ __device__
inline T saturate_cast(int16_t v) { return static_cast<T>(v); }

template <typename T>
__host__ __device__
inline T saturate_cast(int32_t v) { return static_cast<T>(v); }

template <typename T>
__host__ __device__
inline T saturate_cast(int64_t v) { return static_cast<T>(v); }

template <typename T>
__host__ __device__
inline T saturate_cast(uint8_t v) { return static_cast<T>(v); }

template <typename T>
__host__ __device__
inline T saturate_cast(uint16_t v) { return static_cast<T>(v); }

template <typename T>
__host__ __device__
inline T saturate_cast(uint32_t v) { return static_cast<T>(v); }

template <typename T>
__host__ __device__
inline T saturate_cast(uint64_t v) { return static_cast<T>(v); }

}  // namespace imgdistort

#include <imgdistort/saturate_cast-inl.h>

#endif  // IMGDISTORT_SATURATE_CAST_H_
