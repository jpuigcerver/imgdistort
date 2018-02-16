void imgdistort_pytorch_gpu_dilate_nchw_f32(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaTensor* input, THCudaTensor* output);

void imgdistort_pytorch_gpu_dilate_nchw_f64(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaDoubleTensor* input, THCudaDoubleTensor* output);

void imgdistort_pytorch_gpu_dilate_nchw_s8(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaCharTensor* input, THCudaCharTensor* output);

void imgdistort_pytorch_gpu_dilate_nchw_s16(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaShortTensor* input, THCudaShortTensor* output);

void imgdistort_pytorch_gpu_dilate_nchw_s32(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaIntTensor* input, THCudaIntTensor* output);

void imgdistort_pytorch_gpu_dilate_nchw_s64(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaLongTensor* input, THCudaLongTensor* output);

void imgdistort_pytorch_gpu_dilate_nchw_u8(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaByteTensor* input, THCudaByteTensor* output);

void imgdistort_pytorch_gpu_erode_nchw_f32(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaTensor* input, THCudaTensor* output);

void imgdistort_pytorch_gpu_erode_nchw_f64(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaDoubleTensor* input, THCudaDoubleTensor* output);

void imgdistort_pytorch_gpu_erode_nchw_s8(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaCharTensor* input, THCudaCharTensor* output);

void imgdistort_pytorch_gpu_erode_nchw_s16(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaShortTensor* input, THCudaShortTensor* output);

void imgdistort_pytorch_gpu_erode_nchw_s32(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaIntTensor* input, THCudaIntTensor* output);

void imgdistort_pytorch_gpu_erode_nchw_s64(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaLongTensor* input, THCudaLongTensor* output);

void imgdistort_pytorch_gpu_erode_nchw_u8(
    const THCudaLongTensor* kernel_sizes, const THCudaByteTensor* kernels,
    const THCudaByteTensor* input, THCudaByteTensor* output);
