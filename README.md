# imgdistort

Library to perform image distortions on the GPU.

## Implemented distortions
- [Affine transformations](https://en.wikipedia.org/wiki/Affine_transformation)
  + Scaling, translation, rotation, shearing, etc.
- Morphological operations
  + [Grayscale erosion](https://en.wikipedia.org/wiki/Erosion_(morphology))
  + [Grayscale dilation](https://en.wikipedia.org/wiki/Dilation_(morphology))

## Image format

Images are represented as continuous floating point (float or double) arrays. 
The library is designed to work with batched images, i.e. processing multiple images simultaneously. 
The layout for each batch of images is: Batch size x Channels x Height x Width (whic is the standard 
layout for batched images in [Torch](http://torch.ch/)).

It is important to keep in mind that the output images have the same size as the original images, 
regardless of the applied operation. That means that you may "lose" part of your input image when 
certain transformations are applied (i.e. affine transformations). To avoid that, pad your images
conveniently.

Additionally, all batched images must have the same size, so if your images have different sizes 
you will also need to pad them.

## TODO

- Write tests
- Add CPU implementation
- Add additional distortions
  + Local Elastic Deformations
  + Pinch
  + Scratches
  + Pixel permutation
  + Motion blur
  + Gaussian blur
  + JPEG compression
  + More ideas: Deep Self-Taught Learning for Handwritten Character Recognition, F. Bastien et al.
