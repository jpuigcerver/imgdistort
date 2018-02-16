#include <imgdistort/gpu/affine_base_test.h>

#include <imgdistort/logging.h>

namespace imgdistort {
namespace gpu {
namespace testing {

template<typename T>
void OriginalTensor(
    const int N, const int C, thrust::device_vector<T> *output) {
  CHECK_NOTNULL(output);
  output->resize(N * C * 16 * 32);
  for (size_t s = 0; s < output->size();) {
    const size_t n = std::min(OriginalImage<T>().size(), output->size() - s);
    thrust::copy_n(OriginalImage<T>().data(), n, output->data() + s);
    s += n;
  }
}

template<typename T>
const thrust::device_vector<T> &OriginalImage() {
  static const thrust::device_vector<T> img_gpu(cpu::testing::OriginalImage<T>());
  return img_gpu;
}

template<typename T>
const thrust::device_vector<T> &Affine1Image() {
  static const thrust::device_vector<T> img_gpu(cpu::testing::Affine1Image<T>());
  return img_gpu;
}

template<typename T>
const thrust::device_vector<T> &Affine2Image() {
  static const thrust::device_vector<T> img_gpu(cpu::testing::Affine2Image<T>());
  return img_gpu;
}

template void OriginalTensor<int8_t>(
    const int, const int, thrust::device_vector<int8_t> *);

template void OriginalTensor<int16_t>(
    const int, const int, thrust::device_vector<int16_t> *);

template void OriginalTensor<int32_t>(
    const int, const int, thrust::device_vector<int32_t> *);

template void OriginalTensor<int64_t>(
    const int, const int, thrust::device_vector<int64_t> *);

template void OriginalTensor<uint8_t>(
    const int, const int, thrust::device_vector<uint8_t> *);

template void OriginalTensor<uint16_t>(
    const int, const int, thrust::device_vector<uint16_t> *);

template void OriginalTensor<uint32_t>(
    const int, const int, thrust::device_vector<uint32_t> *);

template void OriginalTensor<uint64_t>(
    const int, const int, thrust::device_vector<uint64_t> *);

template void OriginalTensor<float>(
    const int, const int, thrust::device_vector<float> *);

template void OriginalTensor<double>(
    const int, const int, thrust::device_vector<double> *);

template const thrust::device_vector<int8_t> &OriginalImage<int8_t>();

template const thrust::device_vector<int16_t> &OriginalImage<int16_t>();

template const thrust::device_vector<int32_t> &OriginalImage<int32_t>();

template const thrust::device_vector<int64_t> &OriginalImage<int64_t>();

template const thrust::device_vector<uint8_t> &OriginalImage<uint8_t>();

template const thrust::device_vector<uint16_t> &OriginalImage<uint16_t>();

template const thrust::device_vector<uint32_t> &OriginalImage<uint32_t>();

template const thrust::device_vector<uint64_t> &OriginalImage<uint64_t>();

template const thrust::device_vector<float> &OriginalImage<float>();

template const thrust::device_vector<double> &OriginalImage<double>();

template const thrust::device_vector<int8_t> &Affine1Image<int8_t>();

template const thrust::device_vector<int16_t> &Affine1Image<int16_t>();

template const thrust::device_vector<int32_t> &Affine1Image<int32_t>();

template const thrust::device_vector<int64_t> &Affine1Image<int64_t>();

template const thrust::device_vector<uint8_t> &Affine1Image<uint8_t>();

template const thrust::device_vector<uint16_t> &Affine1Image<uint16_t>();

template const thrust::device_vector<uint32_t> &Affine1Image<uint32_t>();

template const thrust::device_vector<uint64_t> &Affine1Image<uint64_t>();

template const thrust::device_vector<float> &Affine1Image<float>();

template const thrust::device_vector<double> &Affine1Image<double>();

template const thrust::device_vector<int8_t> &Affine2Image<int8_t>();

template const thrust::device_vector<int16_t> &Affine2Image<int16_t>();

template const thrust::device_vector<int32_t> &Affine2Image<int32_t>();

template const thrust::device_vector<int64_t> &Affine2Image<int64_t>();

template const thrust::device_vector<uint8_t> &Affine2Image<uint8_t>();

template const thrust::device_vector<uint16_t> &Affine2Image<uint16_t>();

template const thrust::device_vector<uint32_t> &Affine2Image<uint32_t>();

template const thrust::device_vector<uint64_t> &Affine2Image<uint64_t>();

template const thrust::device_vector<float> &Affine2Image<float>();

template const thrust::device_vector<double> &Affine2Image<double>();

}  // namespace testing
}  // namespace gpu
}  // namespace imgdistort
