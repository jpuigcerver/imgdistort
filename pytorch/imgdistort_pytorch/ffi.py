from __future__ import absolute_import

import cffi
import torch
from torch.utils.ffi import _wrap_function

from imgdistort_pytorch._imgdistort import lib as _lib, ffi as _ffi
from imgdistort_pytorch import types

# Import symbols in the imgdistort.so lib, and wrap the to call with ffi.
def _import_symbols(loc):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        loc[symbol] = _wrap_function(fn, _ffi)

def _create_impl_dict(loc, func_pattern, types):
    result = {}
    for tensor_suffix, type_desc in types:
        cpu_fn = func_pattern % ('cpu', type_desc)
        gpu_fn = func_pattern % ('gpu', type_desc)
        if cpu_fn in loc:
            result['torch.%s' % tensor_suffix] = loc[cpu_fn]
        if gpu_fn in loc and torch.cuda.is_available():
            result['torch.cuda.%s' % tensor_suffix] = loc[gpu_fn]
    return result

_import_symbols(locals())

AFFINE_DICT = _create_impl_dict(
    locals(), 'imgdistort_pytorch_%s_affine_nchw_%s',
    zip(types.TENSOR_ALL_NAME, types.TENSOR_ALL_DESC))

DILATE_DICT = _create_impl_dict(
    locals(), 'imgdistort_pytorch_%s_dilate_nchw_%s',
    zip(types.TENSOR_ALL_NAME, types.TENSOR_ALL_DESC))

ERODE_DICT = _create_impl_dict(
    locals(), 'imgdistort_pytorch_%s_erode_nchw_%s',
    zip(types.TENSOR_ALL_NAME, types.TENSOR_ALL_DESC))

_gpu_fn = locals().get('imgdistort_pytorch_gpu_affine_nchw_f32', None)

def is_cuda_available():
    return (torch.cuda.is_available() and _gpu_fn is not None)
