void imgdistort_pytorch_cpu_dilate_nchw_f32(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THFloatTensor* input, THFloatTensor* output);

void imgdistort_pytorch_cpu_dilate_nchw_f64(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THDoubleTensor* input, THDoubleTensor* output);

void imgdistort_pytorch_cpu_dilate_nchw_s8(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THCharTensor* input, THCharTensor* output);

void imgdistort_pytorch_cpu_dilate_nchw_s16(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THShortTensor* input, THShortTensor* output);

void imgdistort_pytorch_cpu_dilate_nchw_s32(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THIntTensor* input, THIntTensor* output);

void imgdistort_pytorch_cpu_dilate_nchw_s64(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THLongTensor* input, THLongTensor* output);

void imgdistort_pytorch_cpu_dilate_nchw_u8(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THByteTensor* input, THByteTensor* output);

void imgdistort_pytorch_cpu_erode_nchw_f32(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THFloatTensor* input, THFloatTensor* output);

void imgdistort_pytorch_cpu_erode_nchw_f64(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THDoubleTensor* input, THDoubleTensor* output);

void imgdistort_pytorch_cpu_erode_nchw_s8(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THCharTensor* input, THCharTensor* output);

void imgdistort_pytorch_cpu_erode_nchw_s16(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THShortTensor* input, THShortTensor* output);

void imgdistort_pytorch_cpu_erode_nchw_s32(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THIntTensor* input, THIntTensor* output);

void imgdistort_pytorch_cpu_erode_nchw_s64(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THLongTensor* input, THLongTensor* output);

void imgdistort_pytorch_cpu_erode_nchw_u8(
    const THLongTensor* kernel_sizes, const THByteTensor* kernels,
    const THByteTensor* input, THByteTensor* output);
