#ifndef IMGDISTORT_LOGGING_H_
#define IMGDISTORT_LOGGING_H_

#ifdef WITH_IPP
#include <ipp.h>
#endif

#ifdef WITH_GLOG

#include <glog/logging.h>

#define CHECK_IPP_CALL(status)                                          \
  do {                                                                  \
    const IppStatus status_ = (status);                                 \
    CHECK_EQ(status_, ippStsNoErr)                                      \
        << "IPP call " #status ": "                                     \
        <<  ippGetStatusString(status_);                                \
  } while(0)

#define CHECK_CUDA_CALL(status)                                         \
  do {                                                                  \
    const cudaError_t status_ = (status);                               \
    CHECK_EQ(status_, cudaSuccess)                                      \
        << "CUDA call " #status ": "                                    \
        << cudaGetErrorString(status_);                                 \
  } while(0)

#define CHECK_LAST_CUDA_CALL() CHECK_CUDA_CALL(cudaPeekAtLastError())

#else

#include <cstdio>
#include <cstdlib>

namespace imgdistort {
namespace internal {
template <typename T>
T* check_not_null(const char* file, const int line, const char* msg, T* ptr) {
  if (ptr == nullptr) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
  }
  return ptr;
}
}  // namespace internal
}  // namespace imgdistort

#define CHECK_FMT(file, line, msg, cond)                                \
  do {                                                                  \
    if (!(cond)) {                                                      \
      fprintf(                                                          \
          stderr, "Check failed at file %s, line line %d: %s\n",        \
          file, line, msg);                                             \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while(0)


#define CHECK(cond) CHECK_FMT(__FILE__, __LINE__, "", cond)

#define CHECK_EQ(a, b)                                                  \
  CHECK_FMT(__FILE__, __LINE__,                                         \
            #a " must be equal to " #b " (", (a) == (b))

#define CHECK_NE(a, b)                                                  \
  CHECK_FMT(__FILE__, __LINE__,                                         \
            #a " must be different to " #b, (a) != (b))

#define CHECK_GE(a, b)                                                  \
  CHECK_FMT(__FILE__, __LINE__,                                         \
            #a " must be greater than or equal to " #b, (a) >= (b))

#define CHECK_GT(a, b)                                                  \
  CHECK_FMT(__FILE__, __LINE__,                                         \
            #a " must be greater than " #b, (a) > (b))

#define CHECK_LE(a, b)                                                  \
  CHECK_FMT(__FILE__, __LINE__,                                         \
            #a " must be lower than or equal to " #b, (a) <= (b))

#define CHECK_LT(a, b)                                                  \
  CHECK_FMT(__FILE__, __LINE__,                                         \
            #a " must be lower than " #b, (a) < (b))

#define CHECK_NOTNULL(a)                                                \
  imgdistort::internal::check_not_null(                                 \
      __FILE__, __LINE__, #a " must be non NULL", (a))

#define CHECK_IPP_CALL(status)                                          \
  do {                                                                  \
    const IppStatus status_ = (status);                                 \
    const std::string msg_ =                                            \
        "IPP call: " + ippGetStatusString(status_);                     \
    CHECK_FMT(__FILE__, __LINE__, msg_, status_);                       \
  } while(0)

#define CHECK_CUDA_CALL(status)                                          \
  do {                                                                   \
    const cudaError_t status_ = (status);                                \
    const std::string msg_ =                                             \
        "CUDA call: " + std::string(cudaGetErrorString(status_));        \
    CHECK_FMT(__FILE__, __LINE__, msg_.c_str(), status_ == cudaSuccess); \
  } while(0)

#define CHECK_LAST_CUDA_CALL() CHECK_CUDA_CALL(cudaPeekAtLastError())

#endif  // WITH_GLOG

#endif  // IMGDISTORT_LOGGING_H_
