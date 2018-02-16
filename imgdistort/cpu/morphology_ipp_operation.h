#ifndef IMGDISTORT_MORPHOLOGY_CPU_IPP_OPERATION_H_
#define IMGDISTORT_MORPHOLOGY_CPU_IPP_OPERATION_H_

#include <ipp.h>

#include <cstdint>

namespace imgdistort {
namespace cpu {
namespace internal {

template <typename T>
class MorphologyOperation {
 public:
  MorphologyOperation() : spec_(nullptr), buffer_(nullptr) {}

  virtual ~MorphologyOperation() {
    Destroy();
  }

  void Initialize(const int W, const int Mw, const int Mh, const uint8_t* M) {
    Destroy();

    const IppiSize maskSize{Mw, Mh};
    // Get spec and buffer sizes
    int specSize = 0, bufferSize = 0;
    CHECK_IPP_CALL(BorderGetSize(W, maskSize, &specSize, &bufferSize));
    // Allocate memory for morphology spec and buffer
    spec_ = (IppiMorphState*)ippsMalloc_8u(specSize);
    buffer_ = (Ipp8u*)ippsMalloc_8u(bufferSize);
    // Initialize morphology operation
    CHECK_IPP_CALL(BorderInitialize(W, M, maskSize, spec_, buffer_));
  }

  void Destroy() {
    if (spec_) { ippsFree(spec_); }
    if (buffer_) { ippsFree(buffer_); }
  }

  virtual IppStatus operator()(
      const T* pSrc, int srcStep, T* pDst, int dstStep, IppiSize roiSize,
      const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) = 0;

  IppiMorphState* Spec() { return spec_; }
  Ipp8u* Buffer() { return buffer_; }

 private:
  IppiMorphState* spec_;
  Ipp8u* buffer_;

  inline IppStatus BorderInitialize(
      const int roiWidth, const Ipp8u* pMask, const IppiSize& maskSize,
      IppiMorphState* pMorphSpec, Ipp8u* pBuffer) const;

  inline IppStatus BorderGetSize(
      const int roiWidth, const IppiSize& maskSize, int* pSpecSize,
      int* pBufferSize) const;
};

template <typename T>
class ErodeOperation : public MorphologyOperation<T> {
 public:
  IppStatus operator()(
      const T* pSrc, int srcStep, T* pDst, int dstStep, IppiSize roiSize,
      const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) override;
};

template <typename T>
class DilateOperation : public MorphologyOperation<T> {
 public:
  IppStatus operator()(
      const T* pSrc, int srcStep, T* pDst, int dstStep, IppiSize roiSize,
      const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) override;
};


#define DEFINE_OPERATION_SPECIALIZATION(T, SNAME)                       \
  template <>                                                           \
  inline IppStatus MorphologyOperation<T>::BorderGetSize(               \
      const int roiWidth, const IppiSize& maskSize, int* pSpecSize,     \
      int* pBufferSize) const {                                         \
    return ippiMorphologyBorderGetSize_##SNAME##_C1R(                   \
        roiWidth, maskSize, pSpecSize, pBufferSize);                    \
  }                                                                     \
                                                                        \
  template <>                                                           \
  inline IppStatus MorphologyOperation<T>::BorderInitialize(            \
      const int roiWidth, const Ipp8u* pMask, const IppiSize& maskSize, \
      IppiMorphState* pMorphSpec, Ipp8u* pBuffer) const {               \
    return ippiMorphologyBorderInit_##SNAME##_C1R(                      \
        roiWidth, pMask, maskSize, pMorphSpec, pBuffer);                \
  }                                                                     \
                                                                        \
  template <>                                                           \
  IppStatus DilateOperation<T>::operator()(                             \
      const T* pSrc, int srcStep, T* pDst, int dstStep, IppiSize roiSize, \
      const IppiMorphState* pMorphSpec, Ipp8u* pBuffer) {               \
    return ippiDilateBorder_##SNAME##_C1R(                              \
        pSrc, srcStep, pDst, dstStep, roiSize, ippBorderRepl, 0,        \
        pMorphSpec, pBuffer);                                           \
  }                                                                     \
                                                                        \
  template <>                                                           \
  IppStatus ErodeOperation<T>::operator()(                              \
      const T* pSrc, int srcStep, T* pDst, int dstStep, IppiSize roiSize, \
      const IppiMorphState* pMorphSpec, Ipp8u* pBuffer)  {              \
    return ippiErodeBorder_##SNAME##_C1R(                               \
        pSrc, srcStep, pDst, dstStep, roiSize, ippBorderRepl, 0,        \
        pMorphSpec, pBuffer);                                           \
  }                                                                     \

DEFINE_OPERATION_SPECIALIZATION(float, 32f)
DEFINE_OPERATION_SPECIALIZATION(int16_t, 16s)
DEFINE_OPERATION_SPECIALIZATION(uint8_t, 8u)
DEFINE_OPERATION_SPECIALIZATION(uint16_t, 16u)
#undef DEFINE_OPERATION_SPECIALIZATION

}  // namespace internal
}  // namespace cpu
}  // namespace imgdistort

#endif  // IMGDISTORT_MORPHOLOGY_CPU_IPP_OPERATION_H_
