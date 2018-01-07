import numpy as np
import torch
import unittest

from imgdistort_pytorch import affine

class AffineBaseTestCase(unittest.TestCase):
    def setUp(self):
        self._x = torch.Tensor([ 1, 2, 3, 4, 5, 6, 7, 8,
                                 9,10,11,12,13,14,15,16,
                                 17,18,19,20,21,22,23,24,
                                 25,26,27,28,29,30]).resize_([2, 1, 5, 3])

class IdentityTestCase(AffineBaseTestCase):
    def setUp(self):
        super(IdentityTestCase, self).setUp()
        self._m = torch.Tensor([1,0,0,0,1,0]).resize_([2, 3]).double()

    def convert(self, cuda, dtype):
        self._x = self._x.type(dtype)
        if cuda:
            self._x = self._x.cuda()
            self._m = self._m.cuda()
        else:
            self._x = self._x.cpu()
            self._m = self._m.cpu()

    def run_base_test(self):
        y = affine(self._x, self._m)
        np.testing.assert_array_almost_equal(self._x.cpu(), y.cpu())

        y.zero_()
        affine(self._x, self._m, y)
        np.testing.assert_array_almost_equal(self._x.cpu(), y.cpu())

    def test_cpu_f32(self):
        self.convert(False, 'torch.FloatTensor')
        self.run_base_test()

    def test_cpu_f64(self):
        self.convert(False, 'torch.DoubleTensor')
        self.run_base_test()

    def test_cpu_s16(self):
        self.convert(False, 'torch.ShortTensor')
        self.run_base_test()

    def test_cpu_s32(self):
        self.convert(False, 'torch.IntTensor')
        self.run_base_test()

    def test_cpu_s64(self):
        self.convert(False, 'torch.LongTensor')
        self.run_base_test()

    def test_gpu_f32(self):
        self.convert(True, 'torch.FloatTensor')
        self.run_base_test()

    def test_gpu_f64(self):
        self.convert(True, 'torch.DoubleTensor')
        self.run_base_test()

    def test_gpu_s16(self):
        self.convert(True, 'torch.ShortTensor')
        self.run_base_test()

    def test_gpu_s32(self):
        self.convert(True, 'torch.IntTensor')
        self.run_base_test()

    def test_gpu_s64(self):
        self.convert(True, 'torch.LongTensor')
        self.run_base_test()

if __name__ == '__main__':
    unittest.main()
