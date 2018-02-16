from __future__ import absolute_import

import torch

TENSOR_INT_DESC = ['u8', 's8', 's16', 's32', 's64']
TENSOR_INT_NAME = [
    'ByteTensor',
    'CharTensor',
    'ShortTensor',
    'IntTensor',
    'LongTensor',
]
TENSOR_INT_TYPE = [
    torch.ByteTensor,
    torch.CharTensor,
    torch.ShortTensor,
    torch.IntTensor,
    torch.LongTensor
]

TENSOR_REAL_DESC = ['f32', 'f64']
TENSOR_REAL_NAME = [
    'FloatTensor',
    'DoubleTensor'
]
TENSOR_REAL_TYPE = [
    torch.FloatTensor,
    torch.DoubleTensor
]

TENSOR_ALL_DESC = TENSOR_INT_DESC + TENSOR_REAL_DESC
TENSOR_ALL_NAME = TENSOR_INT_NAME + TENSOR_REAL_NAME
TENSOR_ALL_TYPE = TENSOR_INT_TYPE + TENSOR_REAL_TYPE
