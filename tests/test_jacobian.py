import unittest

import numpy as np
import sympy

from __context__ import src
from src import distortion
from src import jacobian
from src import mathutils as mu


# all tests take 11 sec with evalf(subs=...)
# takes ~7 sec with lambdify
class TestProjectionJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        distortionModel = distortion.DistortionModel.RadialTangential
        cls.jac = jacobian.ProjectionJacobian(distortionModel)
        cls.intrinsicValues = [400, 400, 0, 320, 240, -0.5, 0.2, 0, 0, 0]
        cls.extrinsicValues = [180, 0, 0, 0.1, 0.2, 1.0]
        cls.modelPoint = [0.1, 0.1, 0]

    def test_init(self):
        self.assertEqual(self.jac._intrinsicJacobianBlockExpr.shape, (2, 10))
        self.assertEqual(self.jac._intrinsicJacobianBlockFunctions.shape, (2, 10))
        self.assertEqual(self.jac._intrinsicJacobianBlockInputSymbols.shape, (2, 10))

        self.assertEqual(self.jac._extrinsicJacobianBlockExpr.shape, (2, 6))
        self.assertEqual(self.jac._extrinsicJacobianBlockFunctions.shape, (2, 6))
        self.assertEqual(self.jac._extrinsicJacobianBlockInputSymbols.shape, (2, 6))

    def test_createExpressionIntrinsicProjection(self):
        expr = jacobian.createExpressionIntrinsicProjectionRadTan()
        self.assertNotEqual(str(expr), "None")

    def test__createIntrinsicsJacobianBlock(self):
        intrinsicBlock = self.jac._createIntrinsicsJacobianBlock(self.intrinsicValues,
                self.extrinsicValues, self.modelPoint)
        self.assertEqual(intrinsicBlock.shape, (2, 10))
        self.assertNoNans(intrinsicBlock)
        self.assertNonZero(intrinsicBlock)

    def test__createExtrinsicsJacobianBlock(self):
        extrinsicBlock = self.jac._createExtrinsicsJacobianBlock(self.intrinsicValues,
                self.extrinsicValues, self.modelPoint)
        self.assertEqual(extrinsicBlock.shape, (2, 6))
        self.assertNoNans(extrinsicBlock)
        self.assertNonZero(extrinsicBlock)

    def test_compute(self):
        #self.jac.compute(P,
        pass

    def assertNoNans(self, Q):
        self.assertEqual(np.sum(np.isnan(Q)), 0)

    def assertNonZero(self, Q):
        self.assertGreater(np.sum(np.abs(Q)), 0)


if __name__ == "__main__":
    unittest.main()
