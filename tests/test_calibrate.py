import unittest
import warnings

import numpy as np

from __context__ import src
from src import calibrate
from src import checkerboard
from src import dataset
from src import mathutils as mu
from src import virtualcamera


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

    def testdistort(self):
        k1 = 0.5
        k2 = 0.2
        p1 = 0
        p2 = 0
        k3 = 0
        distortionCoeffients = (k1, k2, p1, p2, k3)

        normalizedPointsNx2 = (self.pointsInWorld / mu.col(self.pointsInWorld[:,2]))[:,:2]
        distortedPoints = calibrate.distort(normalizedPointsNx2, distortionCoeffients)

        self.assertEqual(distortedPoints.shape, normalizedPointsNx2.shape)
        self.assertEqual(normalizedPointsNx2.shape, (distortedPoints.shape[0], 2))

    def testcomputeHomography(self):
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

        Hcomputed = calibrate.computeHomography(x, X)

        self.assertEqual(Hcomputed.shape, (3,3))
        self.assertTrue(np.allclose(Hcomputed, Hexpected))

    def testvecHomog(self):
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

    def testapproximateRotationMatrix(self):
        Q = np.array([
            [0.95, 0, 0],
            [0, 1, -0.05],
            [0, 0, 1.05],
        ])

        R = calibrate.approximateRotationMatrix(Q)

        self.assertAlmostEqual(np.linalg.det(R), 1)

    def testcomputeExtrinsics(self):
        A = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])

        transformsWorldToCamera = calibrate.computeExtrinsics(self.Hs, A)

        self.assertEqual(len(self.Hs), len(transformsWorldToCamera))

    def testcomputeIntrinsicMatrixFrombClosedForm(self):
        A = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])
        b = createbVectorFromIntrinsicMatrix(A)

        Acomputed = calibrate.computeIntrinsicMatrixFrombClosedForm(b)

        self.assertTrue(np.allclose(A, Acomputed))

    def testcomputeIntrinsicMatrixFrombCholesky(self):
        A = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])
        b = createbVectorFromIntrinsicMatrix(A)

        Acomputed = calibrate.computeIntrinsicMatrixFrombCholesky(b)

        self.assertTrue(np.allclose(A, Acomputed))

    def testcomputeIntrinsicMatrix(self):
        Aexpected = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])
        width, height = 640, 480
        distortionVector = (0, 0, 0, 0, 0)
        dataSet = createSyntheticDataset(Aexpected, width, height, distortionVector)
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        Hs = []
        for x, X in allDetections:
            H = calibrate.computeHomography(x, X)
            Hs.append(H)

        A = calibrate.computeIntrinsicMatrix(Hs)

        self.assertTrue(np.allclose(A, Aexpected))


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


def createSyntheticDataset(A, width, height, distortionVector):
    checkerBoard = checkerboard.Checkerboard(9, 6, 0.100)
    virtualCamera = virtualcamera.VirtualCamera(A, distortionVector, width, height)
    numViews = 10
    dataSet = dataset.Dataset(checkerBoard, virtualCamera, numViews)
    return dataSet


if __name__ == "__main__":
    unittest.main()
