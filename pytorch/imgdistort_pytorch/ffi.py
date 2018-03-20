from __future__ import absolute_import

import torch

import imgdistort_pytorch._imgdistort as _imgdistort
from imgdistort_pytorch import types


def _create_impl_dict(func_pattern, types):
    result = {}
    for tensor_suffix, type_desc in types:
        cpu_fn = getattr(_imgdistort, func_pattern % ('cpu', type_desc), None)
        gpu_fn = getattr(_imgdistort, func_pattern % ('gpu', type_desc), None)
        if cpu_fn:
            result['torch.%s' % tensor_suffix] = cpu_fn
        if gpu_fn and torch.cuda.is_available():
            result['torch.cuda.%s' % tensor_suffix] = gpu_fn
    return result


AFFINE_DICT = _create_impl_dict(
    'imgdistort_pytorch_%s_affine_nchw_%s',
    zip(types.TENSOR_ALL_NAME, types.TENSOR_ALL_DESC))

DILATE_DICT = _create_impl_dict(
    'imgdistort_pytorch_%s_dilate_nchw_%s',
    zip(types.TENSOR_ALL_NAME, types.TENSOR_ALL_DESC))

ERODE_DICT = _create_impl_dict(
    'imgdistort_pytorch_%s_erode_nchw_%s',
    zip(types.TENSOR_ALL_NAME, types.TENSOR_ALL_DESC))

_gpu_fn = getattr(_imgdistort, 'imgdistort_pytorch_gpu_affine_nchw_f32', None)


def is_cuda_available():
    return (torch.cuda.is_available() and _gpu_fn is not None)
