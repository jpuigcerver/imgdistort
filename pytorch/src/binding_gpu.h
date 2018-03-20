void imgdistort_gpu_affine_nchw_f32(
    const THCudaDoubleTensor* m, const THCudaTensor* src,
    THCudaTensor* dst, const float border_value);

void imgdistort_gpu_affine_nchw_f64(
    const THCudaDoubleTensor* m, const THCudaDoubleTensor* src,
    THCudaDoubleTensor* dst, const double border_value);

void imgdistort_gpu_affine_nchw_s8(
    const THCudaDoubleTensor* m, const THCudaCharTensor* src,
    THCudaCharTensor* dst, const int8_t border_value);

void imgdistort_gpu_affine_nchw_s16(
    const THCudaDoubleTensor* m, const THCudaShortTensor* src,
    THCudaShortTensor* dst, const int16_t border_value);

void imgdistort_gpu_affine_nchw_s32(
    const THCudaDoubleTensor* m, const THCudaIntTensor* src,
    THCudaIntTensor* dst, const int32_t border_value);

void imgdistort_gpu_affine_nchw_s64(
    const THCudaDoubleTensor* m, const THCudaLongTensor* src,
    THCudaLongTensor* dst, const int64_t border_value);

void imgdistort_gpu_affine_nchw_u8(
    const THCudaDoubleTensor* m, const THCudaByteTensor* src,
    THCudaByteTensor* dst, const uint8_t border_value);



void imgdistort_gpu_dilate_nchw_f32(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaTensor* src, THCudaTensor* dst);

void imgdistort_gpu_dilate_nchw_f64(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaDoubleTensor* src, THCudaDoubleTensor* dst);

void imgdistort_gpu_dilate_nchw_s8(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaCharTensor* src, THCudaCharTensor* dst);

void imgdistort_gpu_dilate_nchw_s16(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaShortTensor* src, THCudaShortTensor* dst);

void imgdistort_gpu_dilate_nchw_s32(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaIntTensor* src, THCudaIntTensor* dst);

void imgdistort_gpu_dilate_nchw_s64(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaLongTensor* src, THCudaLongTensor* dst);

void imgdistort_gpu_dilate_nchw_u8(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaByteTensor* src, THCudaByteTensor* dst);



void imgdistort_gpu_erode_nchw_f32(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaTensor* src, THCudaTensor* dst);

void imgdistort_gpu_erode_nchw_f64(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaDoubleTensor* src, THCudaDoubleTensor* dst);

void imgdistort_gpu_erode_nchw_s8(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaCharTensor* src, THCudaCharTensor* dst);

void imgdistort_gpu_erode_nchw_s16(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaShortTensor* src, THCudaShortTensor* dst);

void imgdistort_gpu_erode_nchw_s32(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaIntTensor* src, THCudaIntTensor* dst);

void imgdistort_gpu_erode_nchw_s64(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaLongTensor* src, THCudaLongTensor* dst);

void imgdistort_gpu_erode_nchw_u8(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaByteTensor* src, THCudaByteTensor* dst);
