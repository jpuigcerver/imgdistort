#ifndef IMGDISTORT_PYTORCH_SRC_BINDING_COMMON_H_
#define IMGDISTORT_PYTORCH_SRC_BINDING_COMMON_H_

#include <cassert>

namespace imgdistort {
namespace pytorch {

template <typename T>
class AffineCaller {
 public:
  virtual void operator()(
      const int N, const int C, const int H, const int W,
      const double* M, const int Mn, const T* src, T* dst,
      const T& border_value) const = 0;
};

template <typename T>
class MorphologyCaller {
 public:
  virtual void operator()(
      const int N, const int C, const int H, const int W,
      const uint8_t* M, const int* Ms, const int Mn,
      const T* src, T* dst) const = 0;
};

template <typename DTYPE, typename MTYPE, typename TTYPE>
inline void imgdistort_affine_nchw(const MTYPE* m, const TTYPE* x, TTYPE* y,
                                   const DTYPE border_value,
                                   const AffineCaller<DTYPE>& caller) {
  assert(m->nDimension == 2 || m->nDimension == 3);
  assert(x->nDimension == 4);
  const int N = x->size[0];
  const int C = x->size[1];
  const int H = x->size[2];
  const int W = x->size[3];
  const int Mn = m->nDimension == 2 ? 1 : m->size[0];
  if (m->nDimension == 2) {
    assert(m->size[0] == 2 && m->size[1] == 3);
  } else {
    assert(m->size[1] == 2 && m->size[2] == 3);
  }

  const double* m_ptr = m->storage->data + m->storageOffset;
  const DTYPE* x_ptr = x->storage->data + x->storageOffset;
  DTYPE* y_ptr = y->storage->data + y->storageOffset;
  caller(N, C, H, W, m_ptr, Mn, x_ptr, y_ptr, border_value);
}

template <typename DTYPE, typename MTYPE, typename TTYPE>
inline void imgdistort_morph_nchw(
    const MTYPE* m, const THIntTensor* ms, const TTYPE* x, TTYPE* y,
    const MorphologyCaller<DTYPE>& caller) {
  assert(ms->nDimension == 2);
  assert(x->nDimension == 4);
  const int N = x->size[0];
  const int C = x->size[1];
  const int H = x->size[2];
  const int W = x->size[3];
  const int Mn = ms->size[0];
  assert(ms->size[1] == 2);

  const int* ms_ptr = ms->storage->data + ms->storageOffset;
  size_t expected_size = 0;
  for (int n = 0; n < Mn; ++n) {
    expected_size += ms_ptr[2 * n + 0] * ms_ptr[2 * n + 1];
  }
  size_t m_size = 1;
  for (int k = 0; k < m->nDimension; ++k) {
    m_size *= m->size[k];
  }
  assert(m_size == expected_size);

  const DTYPE* x_ptr = x->storage->data + x->storageOffset;
  DTYPE* y_ptr = y->storage->data + y->storageOffset;
  const uint8_t* m_ptr = m->storage->data + m->storageOffset;
  caller(N, C, H, W, m_ptr, ms_ptr, Mn, x_ptr, y_ptr);
}

}  // namespace pytorch
}  // namespace imgdistort

#define DEFINE_AFFINE_WRAPPER(DEVICE, TSNAME, DTYPE, MTYPE, TTYPE)      \
  void imgdistort_affine_nchw_##DEVICE##_##TSNAME(                      \
      const MTYPE* m, const TTYPE* x, TTYPE* y, const DTYPE border) {   \
    ::imgdistort::pytorch::imgdistort_affine_nchw<DTYPE, MTYPE, TTYPE>( \
         m, x, y, border,                                               \
         ::imgdistort::pytorch::DEVICE::AffineCaller<DTYPE>());         \
  }

#define DEFINE_MORPHOLOGY_WRAPPER(DEVICE, TSNAME, DTYPE, MTYPE, TTYPE)  \
  void imgdistort_dilate_nchw_##DEVICE##_##TSNAME(                      \
      const MTYPE* m, const THIntTensor* ms, const TTYPE* x,            \
      TTYPE* y) {                                                       \
    ::imgdistort::pytorch::imgdistort_morph_nchw<DTYPE, MTYPE, TTYPE>(  \
         m, ms, x, y,                                                   \
         ::imgdistort::pytorch::DEVICE::DilateCaller<DTYPE>());         \
  }                                                                     \
                                                                        \
  void imgdistort_erode_nchw_##DEVICE##_##TSNAME(                       \
      const MTYPE* m, const THIntTensor* ms, const TTYPE* x,            \
      TTYPE* y) {                                                       \
    ::imgdistort::pytorch::imgdistort_morph_nchw<DTYPE, MTYPE, TTYPE>(  \
         m, ms, x, y,                                                   \
         ::imgdistort::pytorch::DEVICE::ErodeCaller<DTYPE>());          \
  }

#endif  // IMGDISTORT_PYTORCH_SRC_BINDING_COMMON_H_
