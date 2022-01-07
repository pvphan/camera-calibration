import unittest
import warnings

import numpy as np

from __context__ import src
from src import calibrate
from src import dataset
from src import mathutils as mu


class TestCalibrate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pointsInWorld = np.array([
            [1, -1, 0.4, 1],
            [-1, 1, 0.4, 1],
            [0.3, 0.1, 2.0, 1],
            [0.3, -0.1, 2.0, 1],
            [-0.8, 0.4, 1.2, 1],
            [-0.8, 0.2, 1.2, 1],
        ])
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

        A = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])

        width, height = 640, 480
        kExpected = (-0.5, 0.2)
        cls.syntheticDatasetWithoutDistortion = dataset.createSyntheticDataset(A, width, height, kExpected)

    def test_getNormalizationMatrix(self):
        expectedShape = (3,3)
        numPoints = 10
        X = generateRandomPointsInFrontOfCamera(numPoints)

        N_X = calibrate.getNormalizationMatrix(X[:,:2])

        self.assertEqual(N_X.shape, expectedShape)

    def test_estimateHomography(self):
        numPoints = 10
        X = generateRandomPointsInFrontOfCamera(numPoints)
        X[:,2] = 1
        Hexpected = np.array([
            [400, 10, 320],
            [20, 400, 240],
            [0, 0, 1],
        ])
        x = (Hexpected @ X.T).T
        x = (x / mu.col(x[:,2]))[:,:2]

        Hcomputed = calibrate.estimateHomography(x, X[:,:2])

        self.assertEqual(Hcomputed.shape, (3,3))
        self.assertTrue(np.allclose(Hcomputed, Hexpected))

    def test_estimateHomographies(self):
        Aexpected = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])
        width, height = 640, 480
        distortionVector = (0, 0, 0, 0, 0)
        dataSet = dataset.createSyntheticDataset(Aexpected, width, height, distortionVector)
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()

        Hs = calibrate.estimateHomographies(allDetections)

        self.assertEqual(len(Hs), len(allDetections))
        self.assertEqual(Hs[0].shape, (3,3))

    def test_vecHomog(self):
        expectedShape = (1, 6)
        H = np.array([
            [400, 10, 320],
            [20, 400, 240],
            [0, 0, 1],
        ])

        v1 = calibrate.vecHomography(H, 0, 0)
        v2 = calibrate.vecHomography(H, 0, 1)
        v3 = calibrate.vecHomography(H, 1, 1)

        self.assertEqual(v1.shape, expectedShape)
        self.assertEqual(v2.shape, expectedShape)
        self.assertEqual(v3.shape, expectedShape)

    def test_approximateRotationMatrix(self):
        Q = np.array([
            [0.95, 0, 0],
            [0, 1, -0.05],
            [0, 0, 1.05],
        ])

        R = calibrate.approximateRotationMatrix(Q)

        self.assertAlmostEqual(np.linalg.det(R), 1)

    def test_computeExtrinsics(self):
        A = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])

        worldToCameraTransforms = calibrate.computeExtrinsics(self.Hs, A)

        self.assertEqual(len(self.Hs), len(worldToCameraTransforms))

    def test_computeIntrinsicMatrixFrombClosedForm(self):
        Aexpected = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])
        b = createbVectorFromIntrinsicMatrix(Aexpected)

        Acomputed = calibrate.computeIntrinsicMatrixFrombClosedForm(b)

        self.assertTrue(np.allclose(Aexpected, Acomputed))

    def test_computeIntrinsicMatrixFrombCholesky(self):
        Aexpected = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])
        b = createbVectorFromIntrinsicMatrix(Aexpected)

        Acomputed = calibrate.computeIntrinsicMatrixFrombCholesky(b)

        self.assertTrue(np.allclose(Aexpected, Acomputed))

    def test_computeIntrinsicMatrix(self):
        Aexpected = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])
        width, height = 640, 480
        distortionVector = (0, 0, 0, 0, 0)
        dataSet = dataset.createSyntheticDataset(Aexpected, width, height, distortionVector)
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        Hs = calibrate.estimateHomographies(allDetections)

        A = calibrate.computeIntrinsicMatrix(Hs)

        self.assertTrue(np.allclose(A, Aexpected))

    def test_estimateDistortion(self):
        dataSet = self.syntheticDatasetWithoutDistortion
        A = dataSet.getIntrinsicMatrix()
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        allBoardPosesInCamera = dataSet.getAllBoardPosesInCamera()
        kExpected = dataSet.getDistortionVector()

        kComputed = calibrate.estimateDistortion(A, allDetections, allBoardPosesInCamera)

        self.assertTrue(np.allclose(kExpected, kComputed))

    def test_estimateCalibrationParameters(self):
        dataSet = self.syntheticDatasetWithoutDistortion
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()

        Ainitial, Winitial, kInitial = calibrate.estimateCalibrationParameters(allDetections)

        self.assertEqual(Ainitial.shape, (3,3))
        self.assertEqual(len(Winitial), len(allDetections))
        self.assertEqual(len(kInitial), 2)

    def test_composeParameterVector(self):
        dataSet = self.syntheticDatasetWithoutDistortion
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        A, W, k = calibrate.estimateCalibrationParameters(allDetections)

        P = calibrate.composeParameterVector(A, W, k)
        Acomputed, Wcomputed, kComputed = calibrate.decomposeParameterVector(P)

        self.assertEqual(P.shape, (len(W) * 6 + 7, 1))
        self.assertTrue(np.allclose(A, Acomputed))
        self.assertTrue(len(W), len(Wcomputed))
        self.assertTrue(np.allclose(W, Wcomputed))
        self.assertTrue(np.allclose(k, kComputed))

    def test_refineCalibrationParametersSciPy(self):
        dataSet = self.syntheticDatasetWithoutDistortion
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        Aexpected = dataSet.getIntrinsicMatrix()
        kExpected = dataSet.getDistortionVector()
        Wexpected = dataSet.getAllBoardPosesInCamera()
        Ainitial, Winitial, kInitial = calibrate.estimateCalibrationParameters(allDetections)

        #Arefined, Wrefined, kRefined = calibrate.refineCalibrationParametersSciPy(
        #        Ainitial, Winitial, kInitial, allDetections)

        #self.assertTrue(np.allclose(Aexpected, Arefined))
        #self.assertTrue(np.allclose(Wexpected, Wrefined))
        #self.assertTrue(np.allclose(kExpected, kRefined))


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
