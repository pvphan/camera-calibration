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

    def test_compute(self):
        self.jac.compute(P,


if __name__ == "__main__":
    unittest.main()
