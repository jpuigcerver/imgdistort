#ifndef NNUTILS_THW_THTENSOR_H_
#define NNUTILS_THW_THTENSOR_H_

#include <THW/THTraits.h>

#include <vector>

struct THCState;

namespace nnutils {
namespace THW {

template <typename THTensor>
class ConstTensorBase {
 public:
  typedef typename TensorTraits<THTensor>::TType TType;
  typedef typename TensorTraits<THTensor>::DType DType;

  inline const DType* Data() const;

  inline int Dims() const;

  inline long Elems() const;

  inline const long* Size() const;

  inline long Size(const int dim) const;

  inline const long* Stride() const;

  inline long Stride(const int dim) const;

  inline bool IsContiguous() const;

  template <typename OT>
  inline bool IsSameSizeAs(const OT& other) const {
    if (Dims() != other.Dims()) return false;
    for (int d = 0; d < Dims(); ++d) {
      if (Size(d) != other.Size(d)) return false;
    }
    return true;
  }

 protected:
  virtual const TType* GetTensor() const = 0;

  virtual THCState* GetState() const = 0;
};

template <typename THTensor>
class MutableTensorBase : public ConstTensorBase<THTensor> {
 public:
  typedef typename TensorTraits<THTensor>::TType TType;
  typedef typename TensorTraits<THTensor>::DType DType;

  inline DType* Data();

  inline void Fill(const DType& v);

  inline void Resize(const std::vector<long>& sizes) {
    ResizeNd(sizes.size(), sizes.data(), nullptr);
  }

  template <typename OT>
  inline void ResizeAs(const OT& other) {
    if (!ConstTensorBase<THTensor>::IsSameSizeAs(other)) {
      ResizeNd(other.Dims(), other.Size(), other.Stride());
    }
  }

  inline void ResizeNd(int nDimension, const long* size, const long* stride);

  inline void Transpose(int d1, int d2);

  inline void Zero();

 protected:
  virtual TType * GetMutableTensor() = 0;
};

template <typename THTensor>
class ConstTensor : public ConstTensorBase<THTensor> {
 public:
  explicit ConstTensor(const THTensor* tensor) : tensor_(tensor) {}

 protected:
  const THTensor* GetTensor() const override { return tensor_; }

  THCState* GetState() const override { return nullptr; }

  const THTensor* tensor_;
};

template <typename THTensor>
class MutableTensor : public MutableTensorBase<THTensor> {
 public:
  explicit MutableTensor(THTensor* tensor) : tensor_(tensor) {}

 protected:
  const THTensor* GetTensor() const override { return tensor_; }

  THTensor* GetMutableTensor() override { return tensor_; }

  THCState* GetState() const override { return nullptr; }

  THTensor* tensor_;
};

}  // namespace THW
}  // namespace nnutils


#include <THW/generic/THTensor.h>
#include <TH/THGenerateAllTypes.h>

#endif  // NNUTILS_THW_THTENSOR_H_
