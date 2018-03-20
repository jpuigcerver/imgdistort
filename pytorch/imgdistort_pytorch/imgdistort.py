from __future__ import absolute_import

import torch

from imgdistort_pytorch import ffi
from imgdistort_pytorch import utils


class _FunctionBase(object):
    fn_dict = None

    @classmethod
    def Apply(cls, *args, **kwargs):
        # Get tensor type.
        # If a tensor_type keyword is given, use that string. Otherwise,
        # use the type() of the first argument.
        if 'tensor_type' in kwargs:
            tensor_type = kwargs['tensor_type']
            del kwargs['tensor_type']
        else:
            tensor_type = args[0].type()
        # Get function for the tensor type and call it with the args
        fn = cls._fn_dict.get(tensor_type, None)
        assert fn is not None, (
                'Class %s does not support type %s' % (
            cls.__name__, tensor_type))
        return fn(*args, **kwargs)


class _AffineFunction(_FunctionBase):
    _fn_dict = ffi.AFFINE_DICT

    @classmethod
    def Apply(cls, x, affine_matrix, border_value=0, y=None):
        assert torch.is_tensor(x)
        assert torch.is_tensor(affine_matrix)
        assert y is None or torch.is_tensor(y)
        y = y if y is not None else x.clone()
        assert len(x.size()) == 4, 'Tensor x must have 4 dimensions.'
        assert ((affine_matrix.dim() == 3 and
                 affine_matrix.size()[1:] == (2, 3)) or
                (affine_matrix.dim() == 2 and
                 affine_matrix.size() == (2, 3))), (
            'Size of the affine matrix must be (2, 3) or (?, 2, 3).')

        # Make sure that all the tensors are contiguous and
        # on the same device as x.
        x = x.contiguous()
        y = utils.same_device_as(y.contiguous(), x)
        affine_matrix = utils.same_device_as(affine_matrix.contiguous(), x)

        # Call function in the appropriate device
        with torch.cuda.device_of(x):
            super(_AffineFunction, cls).Apply(
                affine_matrix, x, y, border_value,
                tensor_type=x.type())
        return y


class _MorphologyFunction(_FunctionBase):
    @classmethod
    def Apply(cls, x, structuring_element, y=None):
        assert torch.is_tensor(x)
        assert y is None or torch.is_tensor(y)
        y = y if y is not None else x.clone()

        if torch.is_tensor(structuring_element):
            if structuring_element.dim() == 2:
                # There is a single structuring element, get it's size
                structuring_element_size = torch.LongTensor(
                    [structuring_element.size()])
            elif structuring_element.dim() == 3:
                # All structuring elements have the same size
                structuring_element_size = torch.LongTensor(
                    structuring_element.size()[0] *
                    [structuring_element.size()[1:]])
            else:
                raise ValueError(
                    'The structuring element should be a matrix (2d tensor), '
                    'a 3d tensor, or a list of 2d tensors')
        elif isinstance(structuring_element, (list, tuple)):
            structuring_element_size = []
            for i, kern in enumerate(structuring_element):
                assert (torch.is_tensor(kern) and
                        kern.dim() == 2), (
                        ('The %d-th structuring element should be '
                         'a matrix') % i)
                structuring_element_size.append(kern.size())
            structuring_element_size = torch.LongTensor(
                structuring_element_size)

            # Copy all structuring_element to a contiguous array of memory
            structuring_element = [
                kern.view(kern.numel()) for kern in structuring_element]
            structuring_element = torch.cat(structuring_element)
        else:
            raise TypeError(
                'The structuring element should be a matrix (2d tensor), '
                'a 3d tensor, or a list of 2d tensors')

        x = x.contiguous()
        structuring_element_size = utils.same_device_as(
            structuring_element_size, x)
        structuring_element = utils.same_device_as(
            structuring_element.contiguous(), x)
        y = utils.same_device_as(y.contiguous(), x)

        # Call function in the appropriate device
        with torch.cuda.device_of(x):
            super(_MorphologyFunction, cls).Apply(
                structuring_element_size, structuring_element, x, y,
                tensor_type=x.type())
        return y


class _DilateFunction(_MorphologyFunction):
    _fn_dict = ffi.DILATE_DICT

    @classmethod
    def Apply(cls, x, structuring_element, y=None):
        super(_DilateFunction, cls).Apply(x, structuring_element, y)


class _ErodeFunction(_MorphologyFunction):
    _fn_dict = ffi.ERODE_DICT

    @classmethod
    def Apply(cls, x, structuring_element, y=None):
        super(_ErodeFunction, cls).Apply(x, structuring_element, y)


def affine(batch_input, affine_matrix, border_value=0, batch_output=None):
    r"""Apply the affine transformation given by the affine matrix
    (or matrices) to a batch of images.

    Args:
      batch_input: a tensor representing a batch of images using NCHW layout.
      affine_matrix: ? x 2 x 3 DoubleTensor containing the affine matrix.
      border_value: border value, Default: ``0''
      batch_output: output tensor, if not given a new tensor will be created.
          Default: ``None''
    """
    return _AffineFunction.Apply(batch_input, affine_matrix, border_value,
                                 batch_output)


def dilate(batch_input, structuring_element, batch_output=None):
    r"""Apply a morphology dilation to a batch of images, given by a
    structuring element matrix.

    If a list of structuring element matrices is given, each matrix will be
    applied to each image in the batch.

    Args:
      batch_input: a tensor representing a batch of images using NCHW layout.
      structuring_element: matrix or list of matrices representing the
        structuring element for the operation.
      batch_output: output tensor, if not given a new tensor will be created.
          Default: ``None''
    """
    return _DilateFunction.Apply(batch_input, structuring_element,
                                 batch_output)


def erode(batch_input, structuring_element, batch_output=None):
    r"""Apply a morphology erosion to a batch of images, given by a
    structuring element matrix.

    If a list of structuring element matrices is given, each matrix will be
    applied to each image in the batch.

    Args:
      batch_input: a tensor representing a batch of images using NCHW layout.
      structuring_element: matrix or list of matrices representing the
        structuring element for the operation.
      batch_output: output tensor, if not given a new tensor will be created.
          Default: ``None''
    """
    return _ErodeFunction.Apply(batch_input, structuring_element,
                                batch_output)


__all__ = ['affine', 'dilate', 'erode']
