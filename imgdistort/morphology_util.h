#ifndef IMGDISTORT_MORPHOLOGY_UTIL_H_
#define IMGDISTORT_MORPHOLOGY_UTIL_H_

#include <imgdistort/config.h>

namespace imgdistort {
namespace internal {

template <typename T>
struct DilateFunctor {
  __host__ __device__
  static inline const T& f(const T& a, const T& b) {
    return a > b ? a : b;
  }
};

template <typename T>
struct ErodeFunctor {
  __host__ __device__
  static inline const T& f(const T& a, const T& b) {
    return a < b ? a : b;
  }
};

}  // namespace internal
}  // namespace imgdistort

#endif  // IMGDISTORT_MORPHOLOGY_UTIL_H_
