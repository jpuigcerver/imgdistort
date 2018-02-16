void imgdistort_pytorch_cpu_affine_nchw_f32(
    const THDoubleTensor* affine_matrix, const THFloatTensor* input,
    THFloatTensor* output, const float border_value);

void imgdistort_pytorch_cpu_affine_nchw_f64(
    const THDoubleTensor* affine_matrix, const THDoubleTensor* input,
    THDoubleTensor* output, const double border_value);

void imgdistort_pytorch_cpu_affine_nchw_s8(
    const THDoubleTensor* affine_matrix, const THCharTensor* input,
    THCharTensor* output, const int8_t border_value);

void imgdistort_pytorch_cpu_affine_nchw_s16(
    const THDoubleTensor* affine_matrix, const THShortTensor* input,
    THShortTensor* output, const int16_t border_value);

void imgdistort_pytorch_cpu_affine_nchw_s32(
    const THDoubleTensor* affine_matrix, const THIntTensor* input,
    THIntTensor* output, const int32_t border_value);

void imgdistort_pytorch_cpu_affine_nchw_s64(
    const THDoubleTensor* affine_matrix, const THLongTensor* input,
    THLongTensor* output, const int64_t border_value);

void imgdistort_pytorch_cpu_affine_nchw_u8(
    const THDoubleTensor* affine_matrix, const THByteTensor* input,
    THByteTensor* output, const uint8_t border_value);
