import unittest

import numpy as np
import sympy

from __context__ import src
from src import calibrate
from src import dataset
from src import jacobian
from src import mathutils as mu


class TestProjectionJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jac = jacobian.createJacRadTan()
        cls.intrinsicValues = [400, 400, 0, 320, 240, -0.5, 0.2, 0, 0, 0]
        cls.extrinsicValues = [180, 0, 0, 0.1, 0.2, 1.0]
        cls.modelPoints = np.array([
            [0.1, 0.1, 0],
            [0.1, 0.2, 0],
        ])

        width, height = 640, 480
        α, β, γ, uc, vc, k1, k2, p1, p2, k3 = cls.intrinsicValues
        A = np.array([
            [α, γ, uc],
            [0, β, vc],
            [0, 0,  1],
        ])
        k = (k1, k2, p1, p2, k3)
        cls.syntheticDataset = dataset.createSyntheticDataset(A, width, height, k)
        W = cls.syntheticDataset.getAllBoardPosesInCamera()
        cls.P = calibrate.composeParameterVector(A, W, k)

    def test_init(self):
        self.assertEqual(self.jac._intrinsicJacobianBlockExpr.shape, (2, 10))
        self.assertEqual(self.jac._extrinsicJacobianBlockExpr.shape, (2, 6))

    def test__createIntrinsicsJacobianBlock(self):
        intrinsicBlock = self.jac._createIntrinsicsJacobianBlock(self.intrinsicValues,
                self.extrinsicValues, self.modelPoints)
        self.assertEqual(intrinsicBlock.shape, (self.modelPoints.shape[0] * 2, 10))
        self.assertNoNans(intrinsicBlock)
        self.assertNonZero(intrinsicBlock)

    def test__createExtrinsicsJacobianBlock(self):
        extrinsicBlock = self.jac._createExtrinsicsJacobianBlock(self.intrinsicValues,
                self.extrinsicValues, self.modelPoints)
        self.assertEqual(extrinsicBlock.shape, (self.modelPoints.shape[0] * 2, 6))
        self.assertNoNans(extrinsicBlock)
        self.assertNonZero(extrinsicBlock)

    def test_compute(self):
        dataSet = self.syntheticDataset
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        allModelPoints = [modelPoints for sensorPoints, modelPoints in allDetections]
        MN = sum([modelPoints.shape[0] for modelPoints in allModelPoints])
        M = len(allModelPoints)
        L = 10
        K = 6 * M + L

        J = self.jac.compute(self.P, allModelPoints)

        self.assertEqual(J.shape, (2*MN, K))
        self.assertNonZero(J[0, 0])
        self.assertNonZero(J[-1, -1])
        self.assertEqual(J[0, 16], 0)

    def assertNoNans(self, Q):
        self.assertEqual(np.sum(np.isnan(Q)), 0)

    def assertNonZero(self, Q):
        self.assertGreater(np.sum(np.abs(Q)), 0)


class TestEvaluation(unittest.TestCase):
    def test_evaluateBlock(self):
        a, b, c, d, e, f, g, h = sympy.symbols("a b c d e f g h")
        expressionBlock = sympy.Matrix([
            [(a+b+c+d) * e, (a+b*c+d) * f, (a*c*d) * g, a/b/c],
            [(2*b+c**2+d/7) * e, (a/b**4+5-d) * f, (a**b*c**d) * g, a/b**d],
        ], dtype=object)

        orderedSymbols = (a, b, c, d, e, f, g)
        functionBlock = sympy.lambdify(orderedSymbols, expressionBlock, "numpy")

        wP = np.arange(9).reshape(-1, 3)
        X = list(wP[:,0])
        Y = list(wP[:,1])
        Z = list(wP[:,2])
        P = np.array([
            [1, 1, 2, 3],
            [1, 20, 30, 70],
            [1, 1, 2, 3],
        ], dtype=np.float32)
        blockValues = functionBlock(*[P[:,i] for i in range(4)], X, Y, Z)
        blockValuesReshaped = np.moveaxis(blockValues, 2, 0).reshape(-1, blockValues.shape[1])

    def test_createJacobianBlockExpression(self):
        a, b, c, d = sympy.symbols("a b c d")
        uvExpression = np.array([[(a+b+c+d), (2*b+c**2+d/7)]], dtype=object)
        inputSymbols = a, b, c, d
        blockExpression = jacobian.createJacobianBlockExpression(uvExpression, inputSymbols)


if __name__ == "__main__":
    unittest.main()
