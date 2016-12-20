#ifndef IMGDISTORT_DEFINES_H_
#define IMGDISTORT_DEFINES_H_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

#ifdef __cplusplus
namespace imgdistort { typedef enum { CPU = 0, GPU = 1 } DeviceType; }
#endif  // __cplusplus

#define CHECK_IPP_CALL(STATUS)                          \
  do {                                                  \
    const IppStatus status = (STATUS);                  \
    CHECK_EQ(status, ippStsNoErr);                      \
  } while(0)

#define CHECK_NPP_CALL(STATUS)                  \
  do {                                          \
    const NppStatus status = (STATUS);          \
    CHECK_EQ(status, NPP_NO_ERROR);             \
  } while(0)

#endif  // IMGDISTORT_DEFINES_H_
