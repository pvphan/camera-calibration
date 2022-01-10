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
                self.extrinsicValues, self.modelPoints)
        self.assertEqual(intrinsicBlock.shape, (2, 10))
        self.assertNoNans(intrinsicBlock)
        self.assertNonZero(intrinsicBlock)

    def test__createExtrinsicsJacobianBlock(self):
        extrinsicBlock = self.jac._createExtrinsicsJacobianBlock(self.intrinsicValues,
                self.extrinsicValues, self.modelPoints)
        self.assertEqual(extrinsicBlock.shape, (2, 6))
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

        # takes ~14 sec for one iteration
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
        a, b, c, d = sympy.symbols("a b c d")
        expressionBlock = sympy.Matrix([
            [a+b+c+d, a-b-c-d, a*b*c*d, a/b/c/d],
            [a+2*b+c**2+d/7, a/b**4-5*c-d, a**b*c**d, a/b**c/d],
        ], dtype=object)

        functionBlock, inputSymbols = jacobian.createLambdaFunction(expressionBlock)

        valuesDicts = [
            {a: 0, b: 1, c: 2, d: 3},
            {a: -1, b: 20, c: 30, d: 70},
        ]
        blockValues = jacobian.evaluateBlock(functionBlock, inputSymbols, valuesDicts)
        self.assertEqual(blockValues.shape[0], expressionBlock.shape[0] * len(valuesDicts))


if __name__ == "__main__":
    unittest.main()
