#ifndef IMGDISTORT_MORPHOLOGY_BASE_H_
#define IMGDISTORT_MORPHOLOGY_BASE_H_

#define MORPHOLOGY_TEST_IMG_W 25
#define MORPHOLOGY_TEST_IMG_H 20

#include <gtest/gtest.h>
#include "base_test.h"

namespace imgdistort {
namespace testing {

static const TestImage original_image1{
  15, 10, 3,
  "\0\0\263\0\0\0\32\0\0\0\0[q\17\3\0\360\0%\0\0""3\22)\0#\313\36\376\243\0"
  "&V\0\0\0\0\0%,:i!=a\0\0\0\0\0\0\0\0b\0\4""2\31\0k\0\0""3\272\0D\0\0X\\\0"
  ",\0\0\0\0\230\345\0<j\0\11\0\0\0\25\0\37\36\0\0\36\0\0\33\13}\260U\0\0'\377"
  "\33\0""3\0O\216\33\0\0\0\216\240`\0\0%\0\226\0\0\0Z,?\0\0\0\22\337\0\212"
  "\0\0\261\0\0\0\23V\0J9\220\0""6bLV5\0\0>\0\0\200\0\0L\0\4\0\2051ESu\231R"
  "\0\0CY\377\226\0""1\0\203\0\213\0\6\0\312\0\0\0\0\0u\32i\0\226M\0\342\0\0"
  "q\224a\0+5\275/\0\0\0\205\377\0\377\36'\0\0\17\37""5\0\0\0\0\0\314\0\1\0"
  "\0\0)\11:I\0\234\0\20\10\0\0\3m.\260\0""7\0\0\0\0\0\0\335'\0\0\0\0\0\0\24"
  "\0\353\0\0a\37\0\0\34\0\22\323\0""5O=\377\0\261\0\377\377\0\0\0\0o2\224\201"
  "\0\0\0\0\0\0\377\31\0v\0@r\0\0a\0\0\0\0\201\0k\0\0\322\352\0\0\0\31&\0\0"
  "\36\0/&\0""5>\0[\0\0\0\377\227Y\26\0\0\274\0\0\0\260\11\0\0\0A\251\0\0\252"
  "\377\17\0\0\0\33%\0\33M\0""1\0F\0\0@\0\0\40\0\235\0L\\\0\0\0\14\0\377\0\246"
  "y:`\211\0\377\0\2217\206\0\0\0\0\0\263\0\0%X\211\0|\0Z\0\0\332\231\34\0t"
  "i\361w\0""6\7\377\13\10\213\0-\3778\0""8\0\317\15\15\26",
};

static const TestImage original_image2{
  11, 13, 3,
  "\207[\0\265\0*\377\0\24\0F\0\0K\0\0t\0\0\6IA\0\0\0\0\256\2645\30n\262\275"
  "\255\0\225\377[\0\0\0n-\0\0\0^\0\0\355\0\0A\0s-\216\225\0\177h+\0\0\0\0\271"
  "\0\205\333#\0\0\0\2\0\0xe\0\0\27\3774\0;\0\0\0\233F\0\0\31xZ#\0\0'kv\34\0"
  ")t\0L\335\40\0j\0:\"\0\0e\0\0\0\377\14\0\0\26\10\0o\0\0I\20\0s0\37\0\0\0"
  "\220\0\222\377\371\274\0\0\20\\!\0\0\0\0\203>0\212\0\0\36\0\0\0Q\0L_\255"
  "\0\0\0""4\0QJwL\377\\<\0\0\267\0\0\257\241*\377\0l\0'\0""4\362\0\21\0\0["
  "'\0\0\212\0\"\0""6\312\34\206\0\21\0""1\0\0\0\14\0\377\0\0\0\353n\25{\201"
  "\220\0\0I\377\323\0\0\0\0\0,\233\0\0\0~O%\0\0\0J\0\0\356\0P|\0\0n\0\0\0""1"
  "\0\0\11\0\0v\0\207\256QW\0\0\0\0\22\0\377\0\0""4\0`\0\0Ei\0\0\0\301\310\377"
  "\377\0\0""4\33\0,\234\0\0!\0\0*\0\0\0$\0\0""8\0\0\0\0\0\16\0\1\377\0\17z"
  "+\0\377O\0\201\3\0F\0\0\0\20\0""6\254\0#.\0\0e\0\33r;\0\27\0\26\0!\0h\0\0"
  "\0\0\330\0\0X\0D\0\14\0M\225%\32\0u\0\0\0G\0%\0.\0\13\0X\0\377%=\0\0\0(["
  "\341\0$\0\0\0\252\324\4F\334\0E\310r\260y;\0""2\245",
};





template <class I>
class MorphologyTest : public ::testing::Test { };
TYPED_TEST_CASE_P(MorphologyTest);


TYPED_TEST_P(MorphologyTest, Idempotent) {
  typedef TypeParam Interface;
  typedef typename Interface::Allocator Allocator;
  typedef typename Interface::Type T;

  const int S[] = {1, 3, 5, 7};
  const int N = 4, C = 3, H = 29, W = 31;
  for (int i = 0; i < 4; ++i) {
    const int Ms[] = {S[i], 2 * S[i] + 1};
    std::vector<uint8_t> m(Ms[0] * Ms[1]); m[Ms[0] * Ms[1] / 2] = 1;
    const uint8_t* mask = Allocator::CloneToDevice(m);
    const T* src = Allocator::template GenerateRandom<T>(N * C * H * W);
    T* dst = Allocator::template Allocate<T>(N * C * H * W);
    Interface::Dilate(N, C, H, W, mask, Ms, 1, src, W, dst, W);
    {
      const std::vector<T> src_host =
          Allocator::CloneToHostVec(N * C * H * W, src);
      const std::vector<T> dst_host =
          Allocator::CloneToHostVec(N * C * H * W, dst);
      ASSERT_EQ(dst_host, src_host);
    }
    Interface::Erode(N, C, H, W, mask, Ms, 1, src, W, dst, W);
    {
      const std::vector<T> src_host =
          Allocator::CloneToHostVec(N * C * H * W, src);
      const std::vector<T> dst_host =
          Allocator::CloneToHostVec(N * C * H * W, dst);
      ASSERT_EQ(dst_host, src_host);
    }
    Allocator::Deallocate(mask);
    Allocator::Deallocate(src);
    Allocator::Deallocate(dst);
  }
}

REGISTER_TYPED_TEST_CASE_P(MorphologyTest, Idempotent);

}  // namespace testing
}  // namespace imgdistort

#endif  // IMGDISTORT_MORPHOLOGY_BASE_H_
