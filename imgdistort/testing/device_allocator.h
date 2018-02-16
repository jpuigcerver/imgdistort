#ifndef IMGDISTORT_TESTING_DEVICE_ALLOCATOR_H_
#define IMGDISTORT_TESTING_DEVICE_ALLOCATOR_H_

namespace imgdistort {
namespace testing {

enum DeviceType {CPU = 0, GPU = 1};

template <DeviceType>
class DeviceAllocator {
 public:
  template <typename T>
  static T* Allocate(size_t n);

  template <typename T>
  static void Deallocate(T* ptr);

  template <typename T>
  static void CopyToDevice(size_t n, const T* src, T* dst);

  template <typename T>
  static void CopyToHost(size_t n, const T* src, T* dst);

  template <typename T>
  static T* GenerateRandom(size_t n);

  template <typename T>
  static T* CloneToDevice(size_t n, const T* host_ptr) {
    T* dev_ptr = Allocate<T>(n);
    CopyToDevice(n, host_ptr, dev_ptr);
    return dev_ptr;
  }

  template <typename T>
  static T* CloneToDevice(const std::vector<T>& v) {
    T* dev_ptr = Allocate<T>(v.size());
    CopyToDevice(v.size(), v.data(), dev_ptr);
    return dev_ptr;
  }

  template <typename T>
  static T* CloneToHost(size_t n, const T* dev_ptr) {
    T* host_ptr = new T[n];
    CopyToHost(n, dev_ptr, host_ptr);
    return host_ptr;
  }

  template <typename T>
  static std::vector<T> CloneToHostVec(size_t n, const T* dev_ptr) {
    std::vector<T> host_vec(n);
    CopyToHost(n, dev_ptr, host_vec.data());
    return host_vec;
  }
};

}  // namespace testing
}  // namespace imgdistort

#endif // IMGDISTORT_TESTING_DEVICE_ALLOCATOR_H_
