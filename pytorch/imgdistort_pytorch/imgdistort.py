from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.ffi import _wrap_function
from ._imgdistort import lib as _lib, ffi as _ffi

# Import symbols in the imgdistort.so lib, and wrap the to call with ffi.
def _import_symbols(loc):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        loc[symbol] = _wrap_function(fn, _ffi)
_import_symbols(locals())

# If GPU implementations are not defined in the imgdistort.so, set the symbols
# to None.
for t in ['f32', 'f64', 's8', 's16', 's32', 's64', 'u8']:
    if ('imgdistort_affine_nchw_gpu_%s' % t) not in locals():
        locals()['imgdistort_affine_nchw_gpu_%s' % t] = None

# CPU Implementations
_func_map = {
    'torch.FloatTensor': imgdistort_affine_nchw_cpu_f32,
    'torch.DoubleTensor': imgdistort_affine_nchw_cpu_f64,
    'torch.CharTensor': imgdistort_affine_nchw_cpu_s8,
    'torch.ShortTensor': imgdistort_affine_nchw_cpu_s16,
    'torch.IntTensor': imgdistort_affine_nchw_cpu_s32,
    'torch.LongTensor': imgdistort_affine_nchw_cpu_s64,
    'torch.ByteTensor': imgdistort_affine_nchw_cpu_u8,
}

# GPU Implementations
if torch.cuda.is_available():
    _func_map['torch.cuda.FloatTensor'] = imgdistort_affine_nchw_gpu_f32
    _func_map['torch.cuda.DoubleTensor'] = imgdistort_affine_nchw_gpu_f64
    _func_map['torch.cuda.CharTensor'] = imgdistort_affine_nchw_gpu_s8
    _func_map['torch.cuda.ShortTensor'] = imgdistort_affine_nchw_gpu_s16
    _func_map['torch.cuda.IntTensor'] = imgdistort_affine_nchw_gpu_s32
    _func_map['torch.cuda.LongTensor'] = imgdistort_affine_nchw_gpu_s64
    _func_map['torch.cuda.ByteTensor'] = imgdistort_affine_nchw_gpu_u8


def affine(x, m, y=None):
    assert x is not None and torch.is_tensor(x)
    assert m is not None and torch.is_tensor(x)
    assert y is None or torch.is_tensor(out)
    y = y if y is not None else x.clone()
    assert len(x.size()) == 4, 'Tensor x must have 4 dimensions.'
    assert x.size() == y.size(), 'Tensor x and y must have the same size.'
    assert m.size() == (2, 3), 'Size of affine matrix must be (2, 3).'
    is_cuda = True if x.is_cuda else False
    # Affine matrix must be a double tensor
    if is_cuda:
        assert isinstance(m, torch.cuda.DoubleTensor)
    else:
        assert isinstance(m, torch.DoubleTensor)
    # Get the implementation for the current type and device kind
    f = _func_map.get(x.type(), None)
    if f is None or x.type() != y.type():
        raise NotImplementedError(
            'imgdistort_pytorch.affine is not implemented for types ' +
            '%s and %s' % (x.type(), y.type()))
    # Call in the appropriate device
    with torch.cuda.device_of(x):
        f(m, x, y)
    return y
