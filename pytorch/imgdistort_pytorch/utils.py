from __future__ import absolute_import

import torch

from imgdistort_pytorch.ffi import is_cuda_available

def random_tensor(ttype, cuda=False, size=(2, 3, 4, 5)):
    r"""Generate a random tensor for testing purposes."""
    # Note: uniform_() is only defined for FloatTensor and DoubleTensor.
    x = torch.FloatTensor(*size)
    if ttype == torch.ByteTensor:
        x.uniform_(0, 255)
    else:
        x.uniform_(-100, 100)
    # Convert FloatTensor to the final type.
    x = x.type(ttype)
    return x.cuda() if cuda else x

def register_torch_test(cls, pattern_name, run_method, ttype, tdesc,
                        add_cuda=True, *args):
    r"""Register a test implementation to a given test class.

    Args:
      cls: class
      pattern_name: pattern for the name of the test. It must contain the
        keys {device} and {tdesc} (e.g.: `'test_{device}_simple_{tdesc}'').
      run_method: name of the base method to execute the test
        (e.g.: `'run_base_test'').
      ttype: torch tensor type (e.g.: `torch.FloatTensor').
      tdesc: short description of the tensor type (e.g.: `'f32'').
      add_cuda: add a test for the equivalent CUDA tensor type,
        default: `True'.
    """
    setattr(cls, pattern_name.format(device='cpu', tdesc=tdesc),
            lambda self: getattr(cls, run_method)(self, False, ttype, *args))
    if add_cuda and is_cuda_available():
        setattr(cls, pattern_name.format(device='gpu', tdesc=tdesc),
                lambda self: getattr(cls, run_method)(self, True, ttype, *args))

def same_device_as(x, y):
    r"""Copy x to the same device as y, if necessary."""
    if y.is_cuda:
        return x.cuda(device=y.get_device())
    else:
        return x.cpu()
