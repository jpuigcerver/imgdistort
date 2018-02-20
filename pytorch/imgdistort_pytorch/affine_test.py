from __future__ import absolute_import

import numpy as np
import torch
import unittest

try:
    import cv2 as cv
except:
    cv = None

from imgdistort_pytorch.ffi import is_cuda_available
from imgdistort_pytorch import affine, types, utils

class AffineTest(unittest.TestCase):
    def run_identity(self, cuda, ttype):
        x = utils.random_tensor(ttype, cuda)
        m = torch.DoubleTensor([[1, 0, 0],
                                [0, 1, 0]])
        m = utils.same_device_as(m, x)
        y = affine(x, m)
        np.testing.assert_array_almost_equal(y, x)

    def run_identity_two_matrix(self, cuda, ttype):
        x = utils.random_tensor(ttype, cuda)
        m = torch.DoubleTensor([[[1, 0, 0],
                                 [0, 1, 0]],
                                [[1, 0, 0],
                                 [0, 1, 0]]])
        m = utils.same_device_as(m, x)
        y = affine(x, m)
        np.testing.assert_array_almost_equal(y, x)

    def run_multiple_images(self, cuda, ttype):
        x = utils.random_tensor(ttype, cuda, size=(5, 1, 7, 9))
        m = torch.from_numpy(np.random.randn(5, 2, 3) * 0.1)
        m = m + torch.eye(2, 3).type('torch.DoubleTensor').view(1, 2, 3)
        # Run affine transform on the five images simultaneously.
        m = utils.same_device_as(m, x)
        y = affine(x, m)
        # Run affine transform on the isolated images.
        y2 = torch.cat([affine(x[0, 0].view(1, 1, 7, 9), m[0]),
                        affine(x[1, 0].view(1, 1, 7, 9), m[1]),
                        affine(x[2, 0].view(1, 1, 7, 9), m[2]),
                        affine(x[3, 0].view(1, 1, 7, 9), m[3]),
                        affine(x[4, 0].view(1, 1, 7, 9), m[4])], dim=0)
        # The output should be the same.
        np.testing.assert_array_almost_equal(y, y2)

    def run_multiple_channels(self, cuda, ttype):
        x = utils.random_tensor(ttype, cuda, size=(1, 3, 7, 9))
        m = torch.DoubleTensor([[ 0.9, 0.1, -0.2],
                                [-0.1, 0.8,  0.3]])
        # Run affine transform on the three channels simultaneously.
        m = utils.same_device_as(m, x)
        y = affine(x, m)
        # Run affine transform on the isolated channels.
        y2 = torch.cat([affine(x[0, 0].view(1, 1, 7, 9), m),
                        affine(x[0, 1].view(1, 1, 7, 9), m),
                        affine(x[0, 2].view(1, 1, 7, 9), m)], dim=1)
        # The output should be the same.
        np.testing.assert_array_almost_equal(y, y2)

    def run_opencv(self, cuda, ttype):
        # My implementation should be very similar to OpenCV's.
        x = utils.random_tensor(ttype, cuda, size=(1, 1, 9, 11))
        m = torch.DoubleTensor([[1.2, -0.2, 0.1],
                                [0.3, 0.8, -0.2]])
        m = utils.same_device_as(m, x)
        y = affine(x, m)

        x_np = x[0, 0].cpu().numpy()
        m_np = m.cpu().numpy()
        y_np = y[0, 0].cpu().numpy()

        y_cv = cv.warpAffine(x_np, m_np, (11, 9), flags=cv.INTER_LINEAR)
        # The difference in the pixels should be < 4.
        np.testing.assert_allclose(y_np, y_cv, atol=4)


# Note: torch.CharTensor is missing because PyTorch does not support the
# conversion of a CharTensor to a numpy array.
TENSOR_TYPE = [x for x in types.TENSOR_ALL_TYPE if x != torch.CharTensor]
TENSOR_DESC = [x for x in types.TENSOR_ALL_DESC if x != 's8']

for test_name, run_method in zip(
        ['test_{device}_identity_{tdesc}',
         'test_{device}_identity_two_matrix_{tdesc}',
         'test_{device}_multiple_channels_{tdesc}',
         'test_{device}_multiple_images_{tdesc}'],
        ['run_identity',
         'run_identity_two_matrix',
         'run_multiple_channels',
         'run_multiple_images']):
    for ttype, tdesc in zip(TENSOR_TYPE, TENSOR_DESC):
        utils.register_torch_test(AffineTest, test_name, run_method, ttype, tdesc, add_cuda=True)


if cv is not None:
    # Only floating types are supported by OpenCV
    for ttype, tdesc in zip(types.TENSOR_REAL_TYPE,
                            types.TENSOR_REAL_DESC):
        utils.register_torch_test(AffineTest,
                                  'test_{device}_opencv_{tdesc}', 'run_opencv',
                                  ttype, tdesc)

if __name__ == '__main__':
    unittest.main()
