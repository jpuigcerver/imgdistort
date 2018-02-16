#ifndef IMGDISTORT_TESTING_BASE_H_
#define IMGDISTORT_TESTING_BASE_H_

#include <gtest/gtest.h>
#include <imgdistort/testing/image.h>

namespace imgdistort {
namespace testing {

template <class C>
::testing::AssertionResult assertContainerNear(
    const char* expr1, const char* expr2, const char* eps_expr,
    const C& c1, const C& c2, const typename C::value_type& eps) {
  if (c1.size() != c2.size()) {
    return ::testing::AssertionFailure()
        << "Arrays \"" << expr1 << "\" and \"" << expr2
        << "\" have different sizes : \""
        << expr1 << "\" [" << ::testing::PrintToString(c1.size()) << "] vs \""
        << expr2 << "\" [" << ::testing::PrintToString(c2.size()) << "]";
  }
  int i = 0;
  for (typename C::const_iterator it1 = c1.begin(), it2 = c2.begin();
       it1 != c1.end();
       ++it1, ++it2, ++i) {
    const typename C::value_type diff = fabs(*it1 - *it2);
    if (diff > eps) {
      return ::testing::AssertionFailure()
          << "The max difference between arrays \""
          << expr1 << "\" and \"" << expr2 << "\" is exceeded by element " << i
          << ", where \"" << expr1 << "\" evaluates to "
          << ::testing::PrintToString(*it1)
          << " and \"" << expr2 << "\" evaluates to "
          << ::testing::PrintToString(*it2);
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace testing
}  // namespace imgdistort

#define EXPECT_CONTAINER_NEAR(c1, c2, eps)                              \
  EXPECT_PRED_FORMAT3(imgdistort::testing::assertContainerNear, c1, c2, eps)

#endif  // IMGDISTORT_TESTING_BASE_H_
