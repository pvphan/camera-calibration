import unittest

import numpy as np

from __context__ import src
from src import distortion
from src import jacobian


class TestJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        distortionModel = distortion.DistortionModel.RadialTangential
        cls.jac = jacobian.ProjectionJacobian(distortionModel)

    def test_init(self):
        self.assertEqual(self.jac._intrinsicJacobianBlockExpr.shape, (2, 10))
        self.assertEqual(self.jac._extrinsicJacobianBlockExpr.shape, (2, 6))

    def test_createExpressionIntrinsicProjection(self):
        expr = jacobian.createExpressionIntrinsicProjectionRadTan()
        self.assertNotEqual(str(expr), "None")

    def test__createIntrinsicsJacobianBlock(self):
        intrinsicValues = [400, 400, 0, 320, 240, -0.5, 0.2, 0, 0, 0]
        extrinsicValues = [0, 0, 90, -0.5, 0.2, 0]
        modelPoint = (0.1, 0.3, 0)
        intrinsicBlock = self.jac._createIntrinsicsJacobianBlock(intrinsicValues,
                extrinsicValues, modelPoint)
        self.assertEqual(intrinsicBlock.shape, (2, 10))
        self.assertEqual(np.sum(np.isnan(intrinsicBlock)), 0)

    def test__createExtrinsicsJacobianBlock(self):
        intrinsicValues = [400, 400, 0, 320, 240, -0.5, 0.2, 0, 0, 0]
        extrinsicValues = [0, 0, 90, -0.5, 0.2, 0]
        modelPoint = (0.1, 0.3, 0)
        extrinsicBlock = self.jac._createExtrinsicsJacobianBlock(intrinsicValues,
                extrinsicValues, modelPoint)
        self.assertEqual(extrinsicBlock.shape, (2, 6))
        self.assertEqual(np.sum(np.isnan(extrinsicBlock)), 0)

    def test_compute(self):
        pass
        #self.jac.compute(P,


if __name__ == "__main__":
    unittest.main()
