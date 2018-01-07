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
    if ('imgdistort_dilate_nchw_gpu_%s' % t) not in locals():
        locals()['imgdistort_dilate_nchw_gpu_%s' % t] = None
    if ('imgdistort_erode_nchw_gpu_%s' % t) not in locals():
        locals()['imgdistort_erode_nchw_gpu_%s' % t] = None


# CPU Implementations
_affine = {
    'torch.FloatTensor': imgdistort_affine_nchw_cpu_f32,
    'torch.DoubleTensor': imgdistort_affine_nchw_cpu_f64,
    'torch.CharTensor': imgdistort_affine_nchw_cpu_s8,
    'torch.ShortTensor': imgdistort_affine_nchw_cpu_s16,
    'torch.IntTensor': imgdistort_affine_nchw_cpu_s32,
    'torch.LongTensor': imgdistort_affine_nchw_cpu_s64,
    'torch.ByteTensor': imgdistort_affine_nchw_cpu_u8,
}
_dilate = {
    'torch.FloatTensor': imgdistort_dilate_nchw_cpu_f32,
    'torch.DoubleTensor': imgdistort_dilate_nchw_cpu_f64,
    'torch.CharTensor': imgdistort_dilate_nchw_cpu_s8,
    'torch.ShortTensor': imgdistort_dilate_nchw_cpu_s16,
    'torch.IntTensor': imgdistort_dilate_nchw_cpu_s32,
    'torch.LongTensor': imgdistort_dilate_nchw_cpu_s64,
    'torch.ByteTensor': imgdistort_dilate_nchw_cpu_u8,
}
_erode = {
    'torch.FloatTensor': imgdistort_erode_nchw_cpu_f32,
    'torch.DoubleTensor': imgdistort_erode_nchw_cpu_f64,
    'torch.CharTensor': imgdistort_erode_nchw_cpu_s8,
    'torch.ShortTensor': imgdistort_erode_nchw_cpu_s16,
    'torch.IntTensor': imgdistort_erode_nchw_cpu_s32,
    'torch.LongTensor': imgdistort_erode_nchw_cpu_s64,
    'torch.ByteTensor': imgdistort_erode_nchw_cpu_u8,
}


# GPU Implementations
if torch.cuda.is_available():
    _affine['torch.cuda.FloatTensor'] = imgdistort_affine_nchw_gpu_f32
    _affine['torch.cuda.DoubleTensor'] = imgdistort_affine_nchw_gpu_f64
    _affine['torch.cuda.CharTensor'] = imgdistort_affine_nchw_gpu_s8
    _affine['torch.cuda.ShortTensor'] = imgdistort_affine_nchw_gpu_s16
    _affine['torch.cuda.IntTensor'] = imgdistort_affine_nchw_gpu_s32
    _affine['torch.cuda.LongTensor'] = imgdistort_affine_nchw_gpu_s64
    _affine['torch.cuda.ByteTensor'] = imgdistort_affine_nchw_gpu_u8
    _dilate['torch.cuda.FloatTensor'] = imgdistort_dilate_nchw_gpu_f32
    _dilate['torch.cuda.DoubleTensor'] = imgdistort_dilate_nchw_gpu_f64
    _dilate['torch.cuda.CharTensor'] = imgdistort_dilate_nchw_gpu_s8
    _dilate['torch.cuda.ShortTensor'] = imgdistort_dilate_nchw_gpu_s16
    _dilate['torch.cuda.IntTensor'] = imgdistort_dilate_nchw_gpu_s32
    _dilate['torch.cuda.LongTensor'] = imgdistort_dilate_nchw_gpu_s64
    _dilate['torch.cuda.ByteTensor'] = imgdistort_dilate_nchw_gpu_u8
    _erode['torch.cuda.FloatTensor'] = imgdistort_erode_nchw_gpu_f32
    _erode['torch.cuda.DoubleTensor'] = imgdistort_erode_nchw_gpu_f64
    _erode['torch.cuda.CharTensor'] = imgdistort_erode_nchw_gpu_s8
    _erode['torch.cuda.ShortTensor'] = imgdistort_erode_nchw_gpu_s16
    _erode['torch.cuda.IntTensor'] = imgdistort_erode_nchw_gpu_s32
    _erode['torch.cuda.LongTensor'] = imgdistort_erode_nchw_gpu_s64
    _erode['torch.cuda.ByteTensor'] = imgdistort_erode_nchw_gpu_u8


def affine(x, m, y=None):
    assert x is not None and torch.is_tensor(x)
    assert m is not None and torch.is_tensor(x)
    assert y is None or torch.is_tensor(y)
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
    # Get the implementation for the current type and device type
    f = _affine.get(x.type(), None)
    if f is None or x.type() != y.type():
        raise NotImplementedError(
            'imgdistort_pytorch.affine is not implemented for types ' +
            '%s and %s' % (x.type(), y.type()))
    # Make sure that all the tensors are contiguous.
    x = x.contiguous()
    m = m.contiguous()
    y = y.contiguous()
    # Call in the appropriate device
    with torch.cuda.device_of(x):
        f(m, x, y)
    return y


def _morphology(func_name, op_dict, x, k, ks=None, y=None):
    assert x is not None and torch.is_tensor(x)
    assert k is not None and torch.is_tensor(k)
    assert ks is None or torch.is_tensor(ks) or isinstance(ks, (list, tuple))
    assert y is None or torch.is_tensor(y)
    y = y if y is not None else x.clone()
    assert len(x.size()) == 4, 'Tensor x must have 4 dimensions.'
    assert x.size() == y.size(), 'Tensor x and y must have the same size.'
    is_cuda = True if x.is_cuda else False
    # Kernels tensor must be a byte (boolean)
    if is_cuda:
        assert isinstance(k, torch.cuda.ByteTensor)
    else:
        assert isinstance(k, torch.ByteTensor)
    # The kernel size matrix must contain integers in the CPU
    if ks is None:
        assert len(k.size()) == 2 or len(k.size()) == 3
        if len(k.size()) == 2:
            ks = torch.IntTensor(list(k.size())).resize_([1, 2]).cpu()
        else:
            ks = torch.IntTensor(x.size()[0] * [k.size()[1:]]).cpu()
    elif isinstance(ks, (list, tuple)):
        if len(ks) == 2 and isinstance(ks[0], int) and isinstance(ks[1], int):
            ks = torch.IntTensor(ks).resize_([1, 2]).cpu()
        else:
            assert all(map(lambda x: isinstance(x, (list, tuple)) and len(x) == 2, ks))
            ks = torch.IntTensor(ks).cpu()
    else:
        assert ks.size()[1] == 2
    assert isinstance(ks, torch.IntTensor)
    # Get the implementation for the current type and device type
    f = op_dict.get(x.type(), None)
    if f is None or x.type() != y.type():
        raise NotImplementedError(
            'imgdistort_pytorch.%s is not implemented for types ' +
            '%s and %s' % (func_name, x.type(), y.type()))
    # Make sure that all the tensors are contiguous.
    x = x.contiguous()
    k = k.contiguous()
    ks = ks.contiguous()
    y = y.contiguous()
    # Call in the appropriate device
    with torch.cuda.device_of(x):
        f(k, ks, x, y)
    return y


def dilate(x, k, ks=None, y=None):
    return _morphology('dilate', _dilate, x, k, ks, y)


def erode(x, k, ks=None, y=None):
    return _morphology('erode', _erode, x, k, ks, y)
