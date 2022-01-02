import unittest
import warnings

import numpy as np

from __context__ import src
from src import calibrate
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
        cls.world_M_camera = np.eye(4)

    def testproject(self):
        K = np.array([
            [450,   0, 360],
            [  0, 450, 240],
            [  0,   0,   1],
        ], dtype=np.float64)
        pointsInCamera = (np.linalg.inv(self.world_M_camera) @ self.pointsInWorld.T).T
        pointsInCameraNormalized = (pointsInCamera / mu.col(pointsInCamera[:,2]))[:,:3]
        expectedPointsInCamera = (K @ pointsInCameraNormalized.T).T

        computedPointsInCamera = calibrate.project(K, np.eye(4), self.pointsInWorld)
        # x, y values are the same
        self.assertTrue(np.allclose(expectedPointsInCamera[:,:2], computedPointsInCamera[:,:2]))
        # z values are all 1, homogeneous
        self.assertTrue(np.allclose(computedPointsInCamera[:,2], 1))

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

        v1 = calibrate.vecHomog(H, 0, 0)
        v2 = calibrate.vecHomog(H, 0, 1)
        v3 = calibrate.vecHomog(H, 1, 1)

        self.assertEqual(v1.shape, expectedShape)
        self.assertEqual(v2.shape, expectedShape)
        self.assertEqual(v3.shape, expectedShape)

    def testcomputeIntrinsicMatrix(self):
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
            [100, 15, 120],
            [0, 200, 340],
            [0, 0, 1],
        ])
        Hs = [H1, H2, H3]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in double_scalars')
            K = calibrate.computeIntrinsicMatrix(Hs)

        self.assertAlmostEqual(K[1,0], 0)
        self.assertAlmostEqual(K[2,0], 0)
        self.assertAlmostEqual(K[2,1], 0)


def generateRandomPointsInFrontOfCamera(numPoints):
    np.random.seed(0)
    pointsInCamera = np.zeros((numPoints, 3))
    pointsInCamera[:,0] = np.random.uniform(-1, 1, numPoints)
    pointsInCamera[:,1] = np.random.uniform(-1, 1, numPoints)
    pointsInCamera[:,2] = np.random.uniform(0.5, 1.5, numPoints)
    return pointsInCamera


if __name__ == "__main__":
    unittest.main()
