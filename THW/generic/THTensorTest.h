#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THW/generic/THTensorTest.h"
#else

namespace nnutils {
namespace THW {
namespace testing {

template <>
THTensor* THTensor_new<THTensor>(void) {
  return THTensor_(new)();
}

template <>
THTensor* THTensor_newWithSize2d<THTensor>(int s1, int s2) {
  return THTensor_(newWithSize2d)(s1, s2);
}

template <>
void THTensor_free<THTensor>(THTensor* tensor) {
  THTensor_(free)(tensor);
}

}  // namespace testing
}  // namespace THW
}  // namespace nnutils

#endif
