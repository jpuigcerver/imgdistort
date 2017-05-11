#ifndef IMGDISTORT_INTERPOLATION_H_
#define IMGDISTORT_INTERPOLATION_H_

#include <cmath>
#include <imgdistort/saturate_cast.h>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace imgdistort {

template <typename T>
__host__ __device__
inline T pixv(const T* src, const int p, const int h, const int w,
              const int y, const int x, const T padv = 0) {
  return (y >= 0 && y < h && x >= 0 && x < w) ? src[y * p + x] : padv;
}

template <typename T>
__host__ __device__
inline T& pixv(T* dst, const int p, const int y, const int x) {
  return dst[y * p + x];
}

template <typename T>
__host__ __device__
inline T blinterp(const T* src, const int p, const int h, const int w,
                  const double y, const double x, const T padv = 0) {
  const int x1 = static_cast<int>(floor(x));
  const int y1 = static_cast<int>(floor(y));
  const int x2 = x1 + 1;
  const int y2 = y1 + 1;

  double v = 0.0;
  v += pixv(src, p, h, w, y1, x1, padv) * ((x2 - x) * (y2 - y));
  v += pixv(src, p, h, w, y1, x2, padv) * ((x - x1) * (y2 - y));
  v += pixv(src, p, h, w, y2, x1, padv) * ((x2 - x) * (y - y1));
  v += pixv(src, p, h, w, y2, x2, padv) * ((x - x1) * (y - y1));

  return saturate_cast<T>(v);
}

}  // namespace imgdistort

#endif  //  IMGDISTORT_INTERPOLATION_H_
