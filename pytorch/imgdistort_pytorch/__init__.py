import torch
from torch.utils.ffi import _wrap_function
from ._imgdistort import lib as _lib, ffi as _ffi

__all__ = []

def _import_symbols(loc):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        loc[symbol] = _wrap_function(fn, _ffi)
        __all__.append(symbol)
_import_symbols(locals())
