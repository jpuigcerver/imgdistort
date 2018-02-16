#ifndef IMGDISTORT_TESTING_IMAGE_H_
#define IMGDISTORT_TESTING_IMAGE_H_

namespace imgdistort {
namespace testing {

struct Image {
  unsigned int   width;
  unsigned int   height;
  unsigned int   bytes_per_pixel; /* 2:RGB16, 3:RGB, 4:RGBA */
  unsigned char  pixel_data[32 * 16 * 4 + 1];
};

}  // namespace testing
}  // namespace imgdistort

#endif // IMGDISTORT_TESTING_IMAGE_H_
