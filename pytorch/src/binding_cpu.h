void imgdistort_cpu_affine_nchw_f32(
    const THDoubleTensor* m, const THFloatTensor* src, THFloatTensor* dst,
    const float border_value);

void imgdistort_cpu_affine_nchw_f64(
    const THDoubleTensor* m, const THDoubleTensor* src, THDoubleTensor* dst,
    const double border_value);

void imgdistort_cpu_affine_nchw_s8(
    const THDoubleTensor* m, const THCharTensor* src, THCharTensor* dst,
    const int8_t border_value);

void imgdistort_cpu_affine_nchw_s16(
    const THDoubleTensor* m, const THShortTensor* src, THShortTensor* dst,
    const int16_t border_value);

void imgdistort_cpu_affine_nchw_s32(
    const THDoubleTensor* m, const THIntTensor* src, THIntTensor* dst,
    const int32_t border_value);

void imgdistort_cpu_affine_nchw_s64(
    const THDoubleTensor* m, const THLongTensor* src, THLongTensor* dst,
    const int64_t border_value);

void imgdistort_cpu_affine_nchw_u8(
    const THDoubleTensor* m, const THByteTensor* src, THByteTensor* dst,
    const uint8_t border_value);



void imgdistort_cpu_dilate_nchw_f32(
    const THByteTensor* m, const THIntTensor* ms, const THFloatTensor* src,
    THFloatTensor* dst);

void imgdistort_cpu_dilate_nchw_f64(
    const THByteTensor* m, const THIntTensor* ms, const THDoubleTensor* src,
    THDoubleTensor* dst);

void imgdistort_cpu_dilate_nchw_s8(
    const THByteTensor* m, const THIntTensor* ms, const THCharTensor* src,
    THCharTensor* dst);

void imgdistort_cpu_dilate_nchw_s16(
    const THByteTensor* m, const THIntTensor* ms, const THShortTensor* src,
    THShortTensor* dst);

void imgdistort_cpu_dilate_nchw_s32(
    const THByteTensor* m, const THIntTensor* ms, const THIntTensor* src,
    THIntTensor* dst);

void imgdistort_cpu_dilate_nchw_s64(
    const THByteTensor* m, const THIntTensor* ms, const THLongTensor* src,
    THLongTensor* dst);

void imgdistort_cpu_dilate_nchw_u8(
    const THByteTensor* m, const THIntTensor* ms, const THByteTensor* src,
    THByteTensor* dst);



void imgdistort_cpu_erode_nchw_f32(
    const THByteTensor* m, const THIntTensor* ms, const THFloatTensor* src,
    THFloatTensor* dst);

void imgdistort_cpu_erode_nchw_f64(
    const THByteTensor* m, const THIntTensor* ms, const THDoubleTensor* src,
    THDoubleTensor* dst);

void imgdistort_cpu_erode_nchw_s8(
    const THByteTensor* m, const THIntTensor* ms, const THCharTensor* src,
    THCharTensor* dst);

void imgdistort_cpu_erode_nchw_s16(
    const THByteTensor* m, const THIntTensor* ms, const THShortTensor* src,
    THShortTensor* dst);

void imgdistort_cpu_erode_nchw_s32(
    const THByteTensor* m, const THIntTensor* ms, const THIntTensor* src,
    THIntTensor* dst);

void imgdistort_cpu_erode_nchw_s64(
    const THByteTensor* m, const THIntTensor* ms, const THLongTensor* src,
    THLongTensor* dst);

void imgdistort_cpu_erode_nchw_u8(
    const THByteTensor* m, const THIntTensor* ms, const THByteTensor* src,
    THByteTensor* dst);
