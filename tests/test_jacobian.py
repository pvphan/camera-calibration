import unittest

import numpy as np

from __context__ import src
from src import distortion
from src import jacobian
from src import mathutils as mu


class TestJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        distortionModel = distortion.DistortionModel.RadialTangential
        cls.jac = jacobian.ProjectionJacobian(distortionModel)

    def test_createIntrinsicsJacobianBlock(self):
        pass

    def test_createExtrinsicsJacobianBlock(self):
        pass

    def test_createExpressionIntrinsicProjection(self):
        expr = jacobian.createExpressionIntrinsicProjection()
        self.assertNotEqual(str(expr), "None")


if __name__ == "__main__":
    unittest.main()
