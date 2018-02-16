#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <THW/THCTensor.h>
#include <THW/THCTensorTest.h>

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorToCpuTraits;

using nnutils::THW::testing::THTensor_newWithSize2d;
using nnutils::THW::testing::THTensor_free;
using nnutils::THW::testing::THTensor_clone2cpu;
using nnutils::THW::testing::kState;

template <typename THTensor>
class TensorTest : public ::testing::Test {};

typedef ::testing::Types<THCudaByteTensor,
                         THCudaCharTensor,
                         THCudaShortTensor,
                         THCudaIntTensor,
                         THCudaLongTensor,
                         THCudaTensor,
                         THCudaDoubleTensor> TensorTypes;
TYPED_TEST_CASE(TensorTest, TensorTypes);

TYPED_TEST(TensorTest, Constructor) {
  TypeParam* tensor = THTensor_newWithSize2d<TypeParam>(5, 3);
  ConstTensor<TypeParam> ct(tensor, kState);

  EXPECT_EQ(tensor->storage->data + tensor->storageOffset,
            ct.Data());

  EXPECT_EQ(2, ct.Dims());

  EXPECT_EQ(15, ct.Elems());

  EXPECT_EQ(5, ct.Size(0));
  EXPECT_EQ(3, ct.Size(1));

  EXPECT_EQ(3, ct.Stride(0));
  EXPECT_EQ(1, ct.Stride(1));

  THTensor_free(tensor);
}

TYPED_TEST(TensorTest, IsContiguous) {
  TypeParam* tensor = THTensor_newWithSize2d<TypeParam>(5, 3);
  MutableTensor<TypeParam> mt(tensor, kState);

  EXPECT_TRUE(mt.IsContiguous());
  mt.Transpose(0, 1);
  EXPECT_FALSE(mt.IsContiguous());

  THTensor_free(tensor);
}

TYPED_TEST(TensorTest, IsSameSizeAs) {
  TypeParam* tensorA = THTensor_newWithSize2d<TypeParam>(5, 3);
  TypeParam* tensorB = THTensor_newWithSize2d<TypeParam>(1, 3);
  THCudaByteTensor*  tensorC = THTensor_newWithSize2d<THCudaByteTensor>(5, 3);
  THCudaTensor* tensorD = THTensor_newWithSize2d<THCudaTensor>(5, 1);
  ConstTensor<TypeParam> ctA(tensorA, kState);
  ConstTensor<TypeParam> ctB(tensorB, kState);
  ConstTensor<THCudaByteTensor> ctC(tensorC, kState);
  ConstTensor<THCudaTensor> ctD(tensorD, kState);

  EXPECT_TRUE(ctA.IsSameSizeAs(ctA));   // same size, same type
  EXPECT_TRUE(ctA.IsSameSizeAs(ctC));   // same size, diff type

  EXPECT_FALSE(ctA.IsSameSizeAs(ctB));  // diff size, same type
  EXPECT_FALSE(ctA.IsSameSizeAs(ctD));  // diff size, diff type

  THTensor_free(tensorA);
  THTensor_free(tensorB);
  THTensor_free(tensorC);
  THTensor_free(tensorD);
}


TYPED_TEST(TensorTest, Fill) {
  typedef typename MutableTensor<TypeParam>::DType DType;
  typedef typename nnutils::THW::TensorToCpuTraits<TypeParam>::Type
      TypeParamCpu;

  TypeParam* tensor = THTensor_newWithSize2d<TypeParam>(2, 3);
  MutableTensor<TypeParam> mt(tensor, kState);
  mt.Fill(3);

  TypeParamCpu* tensor_cpu = THTensor_clone2cpu<TypeParam>(tensor);
  ConstTensor<TypeParamCpu> ct(tensor_cpu);
  EXPECT_THAT(std::vector<DType>(ct.Data(), ct.Data() + 6), ::testing::Each(3));

  THTensor_free(tensor);
  THTensor_free(tensor_cpu);
}

TYPED_TEST(TensorTest, Resize) {
  TypeParam* tensor = THTensor_newWithSize2d<TypeParam>(2, 3);
  MutableTensor<TypeParam> mt(tensor, kState);

  mt.Resize({4, 3, 2, 1});
  EXPECT_EQ(4, mt.Dims());
  EXPECT_EQ(4, mt.Size(0));
  EXPECT_EQ(3, mt.Size(1));
  EXPECT_EQ(2, mt.Size(2));
  EXPECT_EQ(1, mt.Size(3));

  THTensor_free(tensor);
}

TYPED_TEST(TensorTest, ResizeAs) {
  TypeParam* tensorA = THTensor_newWithSize2d<TypeParam>(5, 3);
  THCudaTensor* tensorB = THTensor_newWithSize2d<THCudaTensor>(1, 1);

  MutableTensor<TypeParam> mtA(tensorA, kState);
  MutableTensor<THCudaTensor> mtB(tensorB, kState);

  mtB.ResizeAs(mtA);
  EXPECT_TRUE(mtB.IsSameSizeAs(mtA));
  EXPECT_TRUE(mtA.IsSameSizeAs(mtB));

  THTensor_free(tensorA);
  THTensor_free(tensorB);
}

TYPED_TEST(TensorTest, Zero) {
  typedef typename MutableTensor<TypeParam>::DType DType;
  typedef typename nnutils::THW::TensorToCpuTraits<TypeParam>::Type
      TypeParamCpu;

  TypeParam* tensor = THTensor_newWithSize2d<TypeParam>(2, 3);
  MutableTensor<TypeParam> mt(tensor, kState);
  mt.Fill(3);
  mt.Zero();


  TypeParamCpu* tensor_cpu = THTensor_clone2cpu<TypeParam>(tensor);
  ConstTensor<TypeParamCpu> ct(tensor_cpu);
  EXPECT_THAT(std::vector<DType>(ct.Data(), ct.Data() + 6), ::testing::Each(0));

  THTensor_free(tensor);
  THTensor_free(tensor_cpu);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  kState = THCState_alloc();
  THCudaInit(kState);
  return RUN_ALL_TESTS();
}
