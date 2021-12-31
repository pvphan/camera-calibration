import unittest

import numpy as np

from calibration import calibrate
from calibration import mathutils as mu


class TestCalibrate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pointsInWorld = np.array([
            [1, -1, 0.4, 1],
            [1, -1, 0.4, 1],
            [0.3, -0.1, 2.0, 1],
            [0.3, -0.1, 2.0, 1],
            [-0.8, 0.4, 1.2, 1],
            [-0.8, 0.4, 1.2, 1],
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


if __name__ == "__main__":
    unittest.main()
