#ifndef IMGDISTORT_INTERPOLATION_H_
#define IMGDISTORT_INTERPOLATION_H_

#include <cmath>
#include <imgdistort/config.h>
#include <imgdistort/saturate_cast.h>

namespace imgdistort {

template <typename T>
__host__ __device__
inline T pixv(const T* src, const int p, const int h, const int w,
              const int y, const int x, const T padv) {
  return (y >= 0 && y < h && x >= 0 && x < w) ? src[y * p + x] : padv;
}

template <typename T>
__host__ __device__
inline const T& pixv(const T* dst, const int p, const int y, const int x) {
  return dst[y * p + x];
}

template <typename T>
__host__ __device__
inline T& pixv(T* dst, const int p, const int y, const int x) {
  return dst[y * p + x];
}

template <typename T>
__host__ __device__
inline T blinterp(const T* src, const int p, const int h, const int w,
                  const double y, const double x, const T padv) {
  const int x1 = static_cast<int>(floor(x));
  const int y1 = static_cast<int>(floor(y));
  const int x2 = x1 + 1;
  const int y2 = y1 + 1;

  const double a = x - floor(x);
  const double b = y - floor(y);

  return saturate_cast<T>(
      (pixv(src, p, h, w, y1, x1, padv) * (1.0 - a) +
       pixv(src, p, h, w, y1, x2, padv) * (      a)) * (1.0 - b) +
      (pixv(src, p, h, w, y2, x1, padv) * (1.0 - a) +
       pixv(src, p, h, w, y2, x2, padv) * (      a)) * (      b));
}

}  // namespace imgdistort

#endif  //  IMGDISTORT_INTERPOLATION_H_