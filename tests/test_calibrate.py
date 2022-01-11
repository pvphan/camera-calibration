import unittest
import warnings
from unittest.mock import MagicMock

import numpy as np

from __context__ import src
from src import calibrate
from src import dataset
from src import distortion
from src import mathutils as mu


class TestCalibrate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        H1 = np.array([
            [400, 10, 320],
            [20, 400, 240],
            [0, 0, 1],
        ])
        H2 = np.array([
            [300, 15, 320],
            [20, 300, 240],
            [0, 0, 1],
        ])
        H3 = np.array([
            [200, 15, 120],
            [0, 200, 340],
            [0, 0, 1],
        ])
        cls.Hs = [H1, H2, H3]

        cls.Aexpected = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])

        width, height = 640, 480
        cls.kExpected = (-0.5, 0.2, 0.07, -0.03, 0.05)
        cls.syntheticDataset = dataset.createSyntheticDataset(cls.Aexpected, width, height, cls.kExpected)
        cls.Wexpected = cls.syntheticDataset.getAllBoardPosesInCamera()
        cls.numIntrinsicParams = 10
        cls.numExtrinsicParamsPerView = 6

        distortionModel = distortion.RadialTangentialModel()
        cls.calibrator = calibrate.Calibrator(distortionModel)

    def test_estimateCalibrationParameters(self):
        dataSet = self.syntheticDataset
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()

        Ainitial, Winitial, kInitial = self.calibrator.estimateCalibrationParameters(allDetections)

        self.assertEqual(Ainitial.shape, (3,3))
        self.assertEqual(len(Winitial), len(allDetections))
        self.assertEqual(len(kInitial), 5)

    def test_composeParameterVector(self):
        dataSet = self.syntheticDataset
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        A = self.Aexpected
        W = self.Wexpected
        k = self.kExpected

        P = calibrate.composeParameterVector(A, W, k)
        Acomputed, Wcomputed, kComputed = calibrate.decomposeParameterVector(P)

        self.assertEqual(P.shape, (len(W) * self.numExtrinsicParamsPerView
                + self.numIntrinsicParams, 1))
        self.assertAllClose(A, Acomputed)
        self.assertTrue(len(W), len(Wcomputed))
        self.assertAllClose(W, Wcomputed)
        self.assertAllClose(k, kComputed)

    def test_refineCalibrationParameters(self):
        dataSet = self.syntheticDataset
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        ydot = calibrate.getSensorPoints(allDetections)
        Ainitial, Winitial, kInitial = self.calibrator.estimateCalibrationParameters(allDetections)
        jac = MagicMock()
        MN = ydot.shape[0]
        K = 10 + len(Winitial) * 6
        J = np.zeros((2*MN, K))
        J[:K,:K] = np.eye(K)
        jac.compute.return_value = J
        maxIters = 1

        sse, Arefined, Wrefined, kRefined = self.calibrator.refineCalibrationParameters(
                Ainitial, Winitial, kInitial, allDetections, jac, maxIters)

        # not checking for correctness, just want it to run
        self.assertIsInstance(sse, float)
        self.assertEqual(Arefined.shape, (3,3))
        self.assertEqual(len(Wrefined), len(allDetections))
        self.assertEqual(len(kRefined), 5)

    def test_projectAllPoints(self):
        dataSet = self.syntheticDataset
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        A, W, k = self.calibrator.estimateCalibrationParameters(allDetections)
        P = calibrate.composeParameterVector(A, W, k)
        allModelPoints = [modelPoints for sensorPoints, modelPoints in allDetections]

        ydot = self.calibrator.projectAllPoints(P, allModelPoints)

        self.assertGreater(ydot.shape[0], 0)
        self.assertEqual(ydot.shape[1], 2)

    def test_getSensorPoints(self):
        dataSet = self.syntheticDataset
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()

        y = calibrate.getSensorPoints(allDetections)

        self.assertGreater(y.shape[0], 0)
        self.assertEqual(y.shape[1], 2)

    def test_computeReprojectionError(self):
        dataSet = self.syntheticDataset
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        Aexpected = dataSet.getIntrinsicMatrix()
        Wexpected = dataSet.getAllBoardPosesInCamera()
        kExpected = dataSet.getDistortionVector()
        Pexpected = calibrate.composeParameterVector(Aexpected, Wexpected, kExpected)

        totalError = self.calibrator._computeReprojectionError(Pexpected, allDetections)

        self.assertAlmostEqual(totalError, 0)

    def assertAllClose(self, A, B, atol=1e-9):
        self.assertTrue(np.allclose(A, B, atol=atol),
                f"\n{A} \n != \n {B}")


def generateRandomPointsInFrontOfCamera(numPoints):
    np.random.seed(0)
    pointsInCamera = np.zeros((numPoints, 3))
    pointsInCamera[:,0] = np.random.uniform(-1, 1, numPoints)
    pointsInCamera[:,1] = np.random.uniform(-1, 1, numPoints)
    pointsInCamera[:,2] = np.random.uniform(0.5, 1.5, numPoints)
    return pointsInCamera


def createbVectorFromIntrinsicMatrix(A):
    """
    From the relation given by Burger eq 88:

        B = (A^-1)^T * A^-1, where B = [B0 B1 B3]
                                       [B1 B2 B4]
                                       [B3 B4 B5]
    """
    Ainv = np.linalg.inv(A)
    B = Ainv.T @ Ainv
    b = (B[0,0], B[0,1], B[1,1], B[0,2], B[1,2], B[2,2])
    return b


if __name__ == "__main__":
    unittest.main()
