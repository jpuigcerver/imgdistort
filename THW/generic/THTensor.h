#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THW/generic/THTensor.h"
#else

namespace nnutils {
namespace THW {

template <> inline
const ConstTensorBase<THTensor>::DType*
ConstTensorBase<THTensor>::Data() const {
  return THTensor_(data)(GetTensor());
}

template <> inline
int ConstTensorBase<THTensor>::Dims() const {
  return THTensor_(nDimension)(GetTensor());
}

template <> inline
bool ConstTensorBase<THTensor>::IsContiguous() const {
  return THTensor_(isContiguous)(GetTensor());
}

template <> inline
long ConstTensorBase<THTensor>::Elems() const {
  return THTensor_(nElement)(GetTensor());
}

template <> inline
const long* ConstTensorBase<THTensor>::Size() const {
  return GetTensor()->size;
}

template <> inline
long ConstTensorBase<THTensor>::Size(int dim) const {
  return THTensor_(size)(GetTensor(), dim);
}

template <> inline
const long* ConstTensorBase<THTensor>::Stride() const {
  return GetTensor()->stride;
}

template <> inline
long ConstTensorBase<THTensor>::Stride(int dim) const {
  return THTensor_(stride)(GetTensor(), dim);
}


template <> inline
MutableTensorBase<THTensor>::DType* MutableTensorBase<THTensor>::Data() {
  return THTensor_(data)(GetMutableTensor());
}

template <> inline
void MutableTensorBase<THTensor>::Fill(
    const MutableTensorBase<THTensor>::DType& v) {
  THTensor_(fill)(GetMutableTensor(), v);
}

template <> inline
void MutableTensorBase<THTensor>::ResizeNd(
    int nDimension, const long* size, const long* stride) {
  THTensor_(resizeNd)(GetMutableTensor(), nDimension,
                      const_cast<long*>(size),
                      const_cast<long*>(stride));
}

template <> inline
void MutableTensorBase<THTensor>::Transpose(int d1, int d2) {
  return THTensor_(transpose)(GetMutableTensor(), nullptr, d1, d2);
}

template <> inline
void MutableTensorBase<THTensor>::Zero() {
  return THTensor_(zero)(GetMutableTensor());
}

}  // namespace THW
}  // namespace nnutils

#endif
