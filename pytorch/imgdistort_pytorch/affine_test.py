from __future__ import absolute_import

import numpy as np
import torch
import unittest

from imgdistort_pytorch import is_cuda_available, affine
from scipy.ndimage.interpolation import affine_transform

def _same_device_as(x, y):
    if y.is_cuda:
        return x.cuda(device=y.get_device())
    else:
        return x.cpu()

class AffineTest(unittest.TestCase):
    def setUp(self):
        pass

    def random_tensor(self, cuda, ttype, size=(2, 3, 4, 5)):
        x = torch.FloatTensor(*size)
        if ttype == torch.ByteTensor:
            x.uniform_(0, 255)
        else:
            x.uniform_(-100, 100)
        x = x.type(ttype)
        if cuda:
            x.cuda()

        return x

    def run_identity(self, cuda, ttype):
        x = self.random_tensor(cuda, ttype)
        m = torch.DoubleTensor([[1, 0, 0],
                                [0, 1, 0]])
        m = _same_device_as(m, x)
        y = affine(x, m)
        np.testing.assert_array_almost_equal(y, x)

    def run_identity_two_matrix(self, cuda, ttype):
        x = self.random_tensor(cuda, ttype)
        m = torch.DoubleTensor([[[1, 0, 0],
                                 [0, 1, 0]],
                                [[1, 0, 0],
                                 [0, 1, 0]]])
        m = _same_device_as(m, x)
        y = affine(x, m)
        np.testing.assert_array_almost_equal(y, x)

    def run_multiple_images(self, cuda, ttype):
        x = self.random_tensor(cuda, ttype, size=(5, 1, 7, 9))
        m = torch.DoubleTensor([[ 0.9, 0.1, -0.2],
                                [-0.1, 0.8,  0.3]])
        # Run affine transform on the five images simultaneously.
        m = _same_device_as(m, x)
        y = affine(x, m)
        # Run affine transform on the isolated images.
        y2 = torch.cat([affine(x[0, 0].view(1, 1, 7, 9), m),
                        affine(x[1, 0].view(1, 1, 7, 9), m),
                        affine(x[2, 0].view(1, 1, 7, 9), m),
                        affine(x[3, 0].view(1, 1, 7, 9), m),
                        affine(x[4, 0].view(1, 1, 7, 9), m)], dim=0)
        # The output should be the same.
        np.testing.assert_array_almost_equal(y, y2)

    def run_multiple_channels(self, cuda, ttype):
        x = self.random_tensor(cuda, ttype, size=(1, 3, 7, 9))
        m = torch.DoubleTensor([[ 0.9, 0.1, -0.2],
                                [-0.1, 0.8,  0.3]])
        # Run affine transform on the three channels simultaneously.
        m = _same_device_as(m, x)
        y = affine(x, m)
        # Run affine transform on the isolated channels.
        y2 = torch.cat([affine(x[0, 0].view(1, 1, 7, 9), m),
                        affine(x[0, 1].view(1, 1, 7, 9), m),
                        affine(x[0, 2].view(1, 1, 7, 9), m)], dim=1)
        # The output should be the same.
        np.testing.assert_array_almost_equal(y, y2)

    def run_scipy(self, cuda, ttype):
        x = self.random_tensor(cuda, ttype, size=(1, 1, 9, 13))
        m = torch.DoubleTensor([[0.8, 0.0, 0.0],
                                [0.0, 0.8, 0.0]])
        m = _same_device_as(m, x)
        y = affine(x, m)

        x_np = x.numpy()
        y_np = y.numpy()
        m_np = m.numpy()

        #y2 = np.zeros(shape=(9, 13), dtype=np.float32)
        #y2 = affine_transform(x_np[0, 0], m_np, order=0, mode='constant', cval=0)
        #print(y2.shape)
        #np.testing.assert_array_almost_equal(y[0, 0], y2)

def register_tests(ttype, tname):
    setattr(AffineTest,
            'test_cpu_identity_%s' % tname,
            lambda self: self.run_identity(False, ttype))
    setattr(AffineTest,
            'test_cpu_identity_two_matrix_%s' % tname,
            lambda self: self.run_identity_two_matrix(False, ttype))
    setattr(AffineTest,
            'test_cpu_multiple_channels_%s' % tname,
            lambda self: self.run_multiple_channels(False, ttype))
    setattr(AffineTest,
            'test_cpu_multiple_images_%s' % tname,
            lambda self: self.run_multiple_images(False, ttype))
    setattr(AffineTest,
            'test_cpu_scipy_%s' % tname,
            lambda self: self.run_scipy(False, ttype))

    if is_cuda_available():
        setattr(AffineTest,
                'test_gpu_identity_%s' % tname,
                lambda self: self.run_identity_two_matrix(True, ttype))
        setattr(AffineTest,
                'test_gpu_identity_two_matrix_%s' % tname,
                lambda self: self.run_identity_two_matrix(True, ttype))
        setattr(AffineTest,
                'test_gpu_multiple_channels_%s' % tname,
                lambda self: self.run_multiple_channels(True, ttype))
        setattr(AffineTest,
                'test_gpu_multiple_images_%s' % tname,
                lambda self: self.run_multiple_images(True, ttype))

# Note: torch.CharTensor is missing because PyTorch does not support the
# conversion of a CharTensor to a numpy array.
TENSOR_TYPES = [
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.ByteTensor,
    torch.ShortTensor,
    torch.IntTensor,
    torch.LongTensor]
TYPE_NAMES = ['f32', 'f64', 'u8', 's16', 's32', 's64']

for ttype, tname in zip(TENSOR_TYPES, TYPE_NAMES):
    register_tests(ttype, tname)

if __name__ == '__main__':
    unittest.main()
