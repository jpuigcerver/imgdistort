void imgdistort_pytorch_gpu_affine_nchw_f32(
    const THCudaDoubleTensor* affine_matrix, const THCudaTensor* input,
    THCudaTensor* output, const float border_value);

void imgdistort_pytorch_gpu_affine_nchw_f64(
    const THCudaDoubleTensor* affine_matrix, const THCudaDoubleTensor* input,
    THCudaDoubleTensor* output, const double border_value);

void imgdistort_pytorch_gpu_affine_nchw_s8(
    const THCudaDoubleTensor* affine_matrix, const THCudaCharTensor* input,
    THCudaCharTensor* output, const int8_t border_value);

void imgdistort_pytorch_gpu_affine_nchw_s16(
    const THCudaDoubleTensor* affine_matrix, const THCudaShortTensor* input,
    THCudaShortTensor* output, const int16_t border_value);

void imgdistort_pytorch_gpu_affine_nchw_s32(
    const THCudaDoubleTensor* affine_matrix, const THCudaIntTensor* input,
    THCudaIntTensor* output, const int32_t border_value);

void imgdistort_pytorch_gpu_affine_nchw_s64(
    const THCudaDoubleTensor* affine_matrix, const THCudaLongTensor* input,
    THCudaLongTensor* output, const int64_t border_value);

void imgdistort_pytorch_gpu_affine_nchw_u8(
    const THCudaDoubleTensor* affine_matrix, const THCudaByteTensor* input,
    THCudaByteTensor* output, const uint8_t border_value);
