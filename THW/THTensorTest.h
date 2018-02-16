#ifndef NNUTILS_THW_THTENSORTEST_H_
#define NNUTILS_THW_THTENSORTEST_H_

#include <THW/THTraits.h>

namespace nnutils {
namespace THW {
namespace testing {

template <typename THTensor>
THTensor* THTensor_new();

template <typename THTensor>
THTensor* THTensor_newWithSize2d(int s1, int s2);

template <typename THTensor>
void THTensor_free(THTensor* tensor);

}  // namespace testing
}  // namespace THW
}  // namespace nnutils

#include <THW/generic/THTensorTest.h>
#include <TH/THGenerateAllTypes.h>

#endif  // NNUTILS_THW_THTENSORTEST_H_
