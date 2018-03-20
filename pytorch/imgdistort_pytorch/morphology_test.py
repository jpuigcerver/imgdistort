from __future__ import absolute_import

import numpy as np
import torch
import unittest

try:
    import cv2 as cv
except:
    cv = None

from imgdistort_pytorch import dilate, erode, types, utils


class MorphologyTest(unittest.TestCase):
    def run_identity(self, cuda, ttype, op):
        x = utils.random_tensor(ttype, cuda)
        m = torch.ByteTensor([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]])
        m = utils.same_device_as(m, x)
        y = op(x, m)
        np.testing.assert_array_almost_equal(y, x)

    def run_multiple_images(self, cuda, ttype, op):
        x = utils.random_tensor(ttype, cuda, size=(5, 1, 7, 9))
        m = torch.from_numpy(np.random.random_integers(0, 1, size=(5, 5, 5)))
        m = m.type('torch.ByteTensor')
        m = utils.same_device_as(m, x)
        # Run the operation on the five images simultaneously.
        y = op(x, m)
        y2 = op(x, [m[0], m[1], m[2], m[3], m[4]])
        # Run the operation on the isolated images.
        y3 = torch.cat([op(x[0, 0].view(1, 1, 7, 9), m[0]),
                        op(x[1, 0].view(1, 1, 7, 9), m[1]),
                        op(x[2, 0].view(1, 1, 7, 9), m[2]),
                        op(x[3, 0].view(1, 1, 7, 9), m[3]),
                        op(x[4, 0].view(1, 1, 7, 9), m[4])], dim=0)
        # The output should be the same.
        np.testing.assert_allclose(y, y2)
        np.testing.assert_allclose(y, y3)

    def run_multiple_channels(self, cuda, ttype, op):
        x = utils.random_tensor(ttype, cuda, size=(1, 3, 7, 9))
        m = torch.from_numpy(np.random.random_integers(0, 1, size=(5, 5)))
        m = m.type('torch.ByteTensor')
        m = utils.same_device_as(m, x)
        # Run the operation on all channels simultaneously.
        y = op(x, m)
        # Run the operation on the isolated images.
        # Run affine transform on the isolated channels.
        y2 = torch.cat([op(x[0, 0].view(1, 1, 7, 9), m),
                        op(x[0, 1].view(1, 1, 7, 9), m),
                        op(x[0, 2].view(1, 1, 7, 9), m)], dim=1)
        # The output should be the same.
        np.testing.assert_allclose(y, y2)

    def run_opencv(self, cuda, ttype, op):
        # My implementation should be very similar to OpenCV's.
        x = utils.random_tensor(ttype, cuda, size=(1, 1, 9, 11))
        m = torch.from_numpy(np.random.random_integers(0, 1, size=(5, 5)))
        m = m.type('torch.ByteTensor')
        m = utils.same_device_as(m, x)
        y = op(x, m)

        x_np = x[0, 0].cpu().numpy()
        m_np = m.cpu().numpy()
        y_np = y[0, 0].cpu().numpy()

        if op == dilate:
            y_cv = cv.dilate(x_np, m_np, borderType=cv.BORDER_REPLICATE)
        else:
            y_cv = cv.erode(x_np, m_np, borderType=cv.BORDER_REPLICATE)

        np.testing.assert_allclose(y_np, y_cv)


# Note: torch.CharTensor is missing because PyTorch does not support the
# conversion of a CharTensor to a numpy array.
TENSOR_TYPE = [x for x in types.TENSOR_ALL_TYPE if x != torch.CharTensor]
TENSOR_DESC = [x for x in types.TENSOR_ALL_DESC if x != 's8']

for test_name, run_method in zip(
        ['{device}_identity_{tdesc}',
         '{device}_multiple_images_{tdesc}',
         '{device}_multiple_channels_{tdesc}'],
        ['run_identity',
         'run_multiple_images',
         'run_multiple_channels']):
    for ttype, tdesc in zip(TENSOR_TYPE, TENSOR_DESC):
        for mname, moper in zip(['dilate', 'erode'], [dilate, erode]):
            utils.register_torch_test(
                MorphologyTest,
                'test_%s_%s' % (mname, test_name),
                run_method, ttype, tdesc, True, moper)

if cv is not None:
    for ttype, tdesc in zip(types.TENSOR_REAL_TYPE + [torch.ByteTensor,
                                                      torch.ShortTensor],
                            types.TENSOR_REAL_DESC + ['u8',
                                                      's16']):
        for mname, moper in zip(['dilate', 'erode'], [dilate, erode]):
            utils.register_torch_test(
                MorphologyTest,
                'test_%s_{device}_opencv_{tdesc}' % mname,
                'run_opencv', ttype, tdesc, True, moper)

if __name__ == '__main__':
    unittest.main()
