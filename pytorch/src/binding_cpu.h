void imgdistort_affine_nchw_cpu_f32(
    const THDoubleTensor* m, const THFloatTensor* src, THFloatTensor* dst);

void imgdistort_affine_nchw_cpu_f64(
    const THDoubleTensor* m, const THDoubleTensor* src, THDoubleTensor* dst);

void imgdistort_affine_nchw_cpu_s8(
    const THDoubleTensor* m, const THCharTensor* src, THCharTensor* dst);

void imgdistort_affine_nchw_cpu_s16(
    const THDoubleTensor* m, const THShortTensor* src, THShortTensor* dst);

void imgdistort_affine_nchw_cpu_s32(
    const THDoubleTensor* m, const THIntTensor* src, THIntTensor* dst);

void imgdistort_affine_nchw_cpu_s64(
    const THDoubleTensor* m, const THLongTensor* src, THLongTensor* dst);

void imgdistort_affine_nchw_cpu_u8(
    const THDoubleTensor* m, const THByteTensor* src, THByteTensor* dst);



void imgdistort_dilate_nchw_cpu_f32(
    const THByteTensor* m, const THIntTensor* ms, const THFloatTensor* src,
    THFloatTensor* dst);

void imgdistort_dilate_nchw_cpu_f64(
    const THByteTensor* m, const THIntTensor* ms, const THDoubleTensor* src,
    THDoubleTensor* dst);

void imgdistort_dilate_nchw_cpu_s8(
    const THByteTensor* m, const THIntTensor* ms, const THCharTensor* src,
    THCharTensor* dst);

void imgdistort_dilate_nchw_cpu_s16(
    const THByteTensor* m, const THIntTensor* ms, const THShortTensor* src,
    THShortTensor* dst);

void imgdistort_dilate_nchw_cpu_s32(
    const THByteTensor* m, const THIntTensor* ms, const THIntTensor* src,
    THIntTensor* dst);

void imgdistort_dilate_nchw_cpu_s64(
    const THByteTensor* m, const THIntTensor* ms, const THLongTensor* src,
    THLongTensor* dst);

void imgdistort_dilate_nchw_cpu_u8(
    const THByteTensor* m, const THIntTensor* ms, const THByteTensor* src,
    THByteTensor* dst);



void imgdistort_erode_nchw_cpu_f32(
    const THByteTensor* m, const THIntTensor* ms, const THFloatTensor* src,
    THFloatTensor* dst);

void imgdistort_erode_nchw_cpu_f64(
    const THByteTensor* m, const THIntTensor* ms, const THDoubleTensor* src,
    THDoubleTensor* dst);

void imgdistort_erode_nchw_cpu_s8(
    const THByteTensor* m, const THIntTensor* ms, const THCharTensor* src,
    THCharTensor* dst);

void imgdistort_erode_nchw_cpu_s16(
    const THByteTensor* m, const THIntTensor* ms, const THShortTensor* src,
    THShortTensor* dst);

void imgdistort_erode_nchw_cpu_s32(
    const THByteTensor* m, const THIntTensor* ms, const THIntTensor* src,
    THIntTensor* dst);

void imgdistort_erode_nchw_cpu_s64(
    const THByteTensor* m, const THIntTensor* ms, const THLongTensor* src,
    THLongTensor* dst);

void imgdistort_erode_nchw_cpu_u8(
    const THByteTensor* m, const THIntTensor* ms, const THByteTensor* src,
    THByteTensor* dst);
