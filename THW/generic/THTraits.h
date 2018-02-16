#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THW/generic/THTraits.h"
#else

namespace nnutils {
namespace THW {

template <>
class TensorTraits<THTensor> {
 public:
  typedef THTensor  TType;
  typedef real      DType;
  typedef real      VType;
};

}  // namespace nnutils
}  // namespace THW

#endif
