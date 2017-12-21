void imgdistort_affine_nchw_gpu_f32(
    const THCudaDoubleTensor* m, const THCudaTensor* src,
    THCudaTensor* dst);

void imgdistort_affine_nchw_gpu_f64(
    const THCudaDoubleTensor* m, const THCudaDoubleTensor* src,
    THCudaDoubleTensor* dst);

void imgdistort_affine_nchw_gpu_s8(
    const THCudaDoubleTensor* m, const THCudaCharTensor* src,
    THCudaCharTensor* dst);

void imgdistort_affine_nchw_gpu_s16(
    const THCudaDoubleTensor* m, const THCudaShortTensor* src,
    THCudaShortTensor* dst);

void imgdistort_affine_nchw_gpu_s32(
    const THCudaDoubleTensor* m, const THCudaIntTensor* src,
    THCudaIntTensor* dst);

void imgdistort_affine_nchw_gpu_s64(
    const THCudaDoubleTensor* m, const THCudaLongTensor* src,
    THCudaLongTensor* dst);

void imgdistort_affine_nchw_gpu_u8(
    const THCudaDoubleTensor* m, const THCudaByteTensor* src,
    THCudaByteTensor* dst);



void imgdistort_dilate_nchw_gpu_f32(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaTensor* src, THCudaTensor* dst);

void imgdistort_dilate_nchw_gpu_f64(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaDoubleTensor* src, THCudaDoubleTensor* dst);

void imgdistort_dilate_nchw_gpu_s8(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaCharTensor* src, THCudaCharTensor* dst);

void imgdistort_dilate_nchw_gpu_s16(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaShortTensor* src, THCudaShortTensor* dst);

void imgdistort_dilate_nchw_gpu_s32(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaIntTensor* src, THCudaIntTensor* dst);

void imgdistort_dilate_nchw_gpu_s64(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaLongTensor* src, THCudaLongTensor* dst);

void imgdistort_dilate_nchw_gpu_u8(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaByteTensor* src, THCudaByteTensor* dst);



void imgdistort_erode_nchw_gpu_f32(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaTensor* src, THCudaTensor* dst);

void imgdistort_erode_nchw_gpu_f64(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaDoubleTensor* src, THCudaDoubleTensor* dst);

void imgdistort_erode_nchw_gpu_s8(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaCharTensor* src, THCudaCharTensor* dst);

void imgdistort_erode_nchw_gpu_s16(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaShortTensor* src, THCudaShortTensor* dst);

void imgdistort_erode_nchw_gpu_s32(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaIntTensor* src, THCudaIntTensor* dst);

void imgdistort_erode_nchw_gpu_s64(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaLongTensor* src, THCudaLongTensor* dst);

void imgdistort_erode_nchw_gpu_u8(
    const THCudaByteTensor* m, const THIntTensor* ms,
    const THCudaByteTensor* src, THCudaByteTensor* dst);
