#ifndef IMGDISTORT_INTERPOLATION_H_
#define IMGDISTORT_INTERPOLATION_H_

#include <cmath>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#define coord_in(x, y, w, h) \
  (((x) < 0 || (y) < 0 || (x) >= (w) || (y) >= (h)) ? 0 : 1)
#define pixv(src, x, y, p) ((src)[(y) * (p) + (x)])

namespace imgdistort {

template <typename T>
__host__ __device__
inline double blinterp(const T* src, double x, double y, int w, int h, int p) {
  const int x_i = static_cast<int>(x);
  const int y_i = static_cast<int>(y);
  const double a = fabs(x - x_i);
  const double b = fabs(y - y_i);
  double n = 0;
  double v = 0.0;
  if (coord_in(x_i    , y_i    , w, h)) {
    v += (1 - a) * (1 - b) * pixv(src, x_i    , y_i    , p);
    n += (1 - a) * (1 - b);
  }
  if (coord_in(x_i + 1, y_i    , w, h)) {
    v += (    a) * (1 - b) * pixv(src, x_i + 1, y_i    , p);
    n += (    a) * (1 - b);
  }
  if (coord_in(x_i    , y_i + 1, w, h)) {
    v += (1 - a) * (    b) * pixv(src, x_i    , y_i + 1, p);
    n += (1 - a) * (    b);

  }
  if (coord_in(x_i + 1, y_i + 1, w, h)) {
    v += (    a) * (    b) * pixv(src, x_i + 1, y_i + 1, p);
    n += (    a) * (    b);
  }
  return n > 0.0 ? v / n : 0.0;
}

}  // namespace imgdistort

#endif  //  IMGDISTORT_INTERPOLATION_H_
