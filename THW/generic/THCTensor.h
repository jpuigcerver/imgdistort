#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THW/generic/THCTensor.h"
#else

namespace nnutils {
namespace THW {

template <> inline
const ConstTensorBase<THCTensor>::DType*
ConstTensorBase<THCTensor>::Data() const {
  return THCTensor_(data)(GetState(), GetTensor());
}

template <> inline
int ConstTensorBase<THCTensor>::Dims() const {
  return THCTensor_(nDimension)(GetState(), GetTensor());
}

template <> inline
bool ConstTensorBase<THCTensor>::IsContiguous() const {
  return THCTensor_(isContiguous)(GetState(), GetTensor());
}

template <> inline
long ConstTensorBase<THCTensor>::Elems() const {
  return THCTensor_(nElement)(GetState(), GetTensor());
}

template <> inline
const long* ConstTensorBase<THCTensor>::Size() const {
  return GetTensor()->size;
}

template <> inline
long ConstTensorBase<THCTensor>::Size(int dim) const {
  return THCTensor_(size)(GetState(), GetTensor(), dim);
}

template <> inline
const long* ConstTensorBase<THCTensor>::Stride() const {
  return GetTensor()->stride;
}

template <> inline
long ConstTensorBase<THCTensor>::Stride(int dim) const {
  return THCTensor_(stride)(GetState(), GetTensor(), dim);
}


template <> inline
MutableTensorBase<THCTensor>::DType* MutableTensorBase<THCTensor>::Data() {
  return THCTensor_(data)(GetState(), GetMutableTensor());
}

template <> inline
void MutableTensorBase<THCTensor>::Fill(
    const MutableTensorBase<THCTensor>::DType& v) {
  THCTensor_(fill)(GetState(), GetMutableTensor(), v);
}

template <> inline
void MutableTensorBase<THCTensor>::ResizeNd(
    int nDimension, const long* size, const long* stride) {
  THCTensor_(resizeNd)(GetState(), GetMutableTensor(), nDimension,
                       const_cast<long*>(size),
                       const_cast<long*>(stride));
}

template <> inline
void MutableTensorBase<THCTensor>::Transpose(int d1, int d2) {
  THCTensor_(transpose)(GetState(), GetMutableTensor(), nullptr, d1, d2);
}

template <> inline
void MutableTensorBase<THCTensor>::Zero() {
  THCTensor_(zero)(GetState(), GetMutableTensor());
}

template <>
class ConstTensor<THCTensor> : public ConstTensorBase<THCTensor> {
 public:
  explicit ConstTensor(const THCTensor* tensor, THCState* state)
      : tensor_(tensor), state_(state) {}

 protected:
  const THCTensor* GetTensor() const override { return tensor_; }

  THCState* GetState() const override { return state_; }

  const THCTensor* tensor_;
  THCState* state_;
};

template <>
class MutableTensor<THCTensor> : public MutableTensorBase<THCTensor> {
 public:
  explicit MutableTensor(THCTensor* tensor, THCState* state)
      : tensor_(tensor), state_(state) {}
  using MutableTensorBase<THCTensor>::Fill;

 protected:
  const THCTensor* GetTensor() const override { return tensor_; }

  THCTensor* GetMutableTensor() override { return tensor_; }

  THCState* GetState() const override { return state_; }

  THCTensor* tensor_;
  THCState* state_;
};

}  // namespace THW
}  // namespace nnutils

#endif
