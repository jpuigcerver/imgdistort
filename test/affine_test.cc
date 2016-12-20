#include <gtest/gtest.h>
#include <imgdistort/affine.h>

#include <Magick++.h>

TEST(AffineTest, AffineTest) {
  try {
    Magick::Image image;
    image.read("lena.png");

    const Magick::Geometry size = image.size();
    const size_t w = size.width(), h = size.height();
    Magick::PixelPacket* pixels = image.getPixels(0, 0, w, h);

    float* src = new float[2 * 3 * h * w];
    float* dst = new float[2 * 3 * h * w];
    for (int i = 0; i < h * w; ++i) {
      src[0 * h * w + i] = pixels[i].red / 65535.0;
      src[1 * h * w + i] = pixels[i].green / 65535.0;
      src[2 * h * w + i] = pixels[i].blue / 65535.0;
    }
    memcpy(src + 3 * h * w, src, sizeof(float) * 3 * h * w);


    const double M[2][2][3] = {
      // img 1
      { {0.3, -0.1, 100},
        {0.2,  0.5, 4} },
      // img2
      { {0.8, 0.0, 100.0},
        {0.0, 0.8, 100.0} }
    };
    imgdistort::affine_nchw<imgdistort::CPU, float>(
        2, 3, h, w, M, 2, src, w, dst, w);

    // Write image 1
    for (int i = 0; i < h * w; ++i) {
      pixels[i].red   = dst[0 * h * w + i] * 65535.0;
      pixels[i].green = dst[1 * h * w + i] * 65535.0;
      pixels[i].blue  = dst[2 * h * w + i] * 65535.0;
    }
    image.syncPixels();
    image.write("lena_out1.png");

    // Write image 2
    for (int i = 0; i < h * w; ++i) {
      pixels[i].red   = dst[3 * h * w + 0 * h * w + i] * 65535.0;
      pixels[i].green = dst[3 * h * w + 1 * h * w + i] * 65535.0;
      pixels[i].blue  = dst[3 * h * w + 2 * h * w + i] * 65535.0;
    }
    image.syncPixels();
    image.write("lena_out2.png");

    delete [] src;
    delete [] dst;

  } catch(Magick::Exception &error) {
    std::cerr << "Caught exception: " << error.what() << std::endl;
  }
}

int main(int argc, char **argv) {
  Magick::InitializeMagick(*argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
