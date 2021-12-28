import unittest

import numpy as np

import mvg


class TestMVG(unittest.TestCase):
    def testcol(self):
        v = mvg.col((1,1,1))
        self.assertEqual(v.shape, (3,1))

    def testskew(self):
        v = mvg.col((0.5, 1.0, -0.25))
        vHatExpected = np.array([
            [0, 0.25, 1.0],
            [-0.25, 0, -0.5],
            [-1.0, 0.5, 0],
        ])

        vHat = mvg.skew(v)

        self.assertEqual(vHat.shape, (3,3))
        self.assertTrue(np.allclose(vHatExpected, vHat))
        self.assertTrue(np.allclose(vHat, -vHat.T))

    def testunskew(self):
        vHat = np.array([
            [0, 0.25, 1.0],
            [-0.25, 0, -0.5],
            [-1.0, 0.5, 0],
        ])
        vExpected = (0.5, 1.0, -0.25)

        v = mvg.unskew(vHat)
        self.assertTrue(np.allclose(vExpected, v))

    def testexp(self):
        z = mvg.col((0, 0, 1))
        theta = 90
        expectedRotation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        computedRotation = mvg.exp(np.radians(theta) * mvg.skew(z))
        self.assertEqual(computedRotation.shape, (3,3))
        self.assertTrue(np.allclose(expectedRotation, computedRotation))

    def testexpZeroRot(self):
        z = mvg.col((0, 0, 1))
        theta = 0
        expectedRotation = np.eye(3)
        computedRotation = mvg.exp(np.radians(theta) * mvg.skew(z))

        self.assertEqual(computedRotation.shape, (3,3))
        self.assertTrue(np.allclose(expectedRotation, computedRotation))

    def testeulerToRotationMatrix1(self):
        expectedRotation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        eulerAngles = (0, 0, 90)

        computedRotation = mvg.eulerToRotationMatrix(eulerAngles)

        self.assertEqual(computedRotation.shape, (3,3))
        self.assertTrue(np.allclose(expectedRotation, computedRotation),
                f"\nexpected:\n{expectedRotation}\n\ncomputed:\n{computedRotation}")

    def testeulerToRotationMatrix2(self):
        expectedRotation = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ])
        eulerAngles = (0, 180, 0)

        computedRotation = mvg.eulerToRotationMatrix(eulerAngles)

        self.assertEqual(computedRotation.shape, (3,3))
        self.assertTrue(np.allclose(expectedRotation, computedRotation),
                f"\nexpected:\n{expectedRotation}\n\ncomputed:\n{computedRotation}")

    def testeulerToRotationMatrix3(self):
        expectedRotation = np.array([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0],
        ])
        eulerAngles = (0, 90, 90)

        computedRotation = mvg.eulerToRotationMatrix(eulerAngles)

        self.assertEqual(computedRotation.shape, (3,3))
        self.assertTrue(np.allclose(expectedRotation, computedRotation),
                f"\nexpected:\n{expectedRotation}\n\ncomputed:\n{computedRotation}")

    def testproject(self):
        pointsInWorld = np.array([
            [1, -1, 0.4, 1],
            [1, -1, 0.4, 1],
            [0.3, -0.1, 2.0, 1],
            [0.3, -0.1, 2.0, 1],
            [-0.8, 0.4, 1.2, 1],
            [-0.8, 0.4, 1.2, 1],
        ])
        world_M_camera = np.eye(4)

        K = np.array([
            [450,   0, 360],
            [  0, 450, 240],
            [  0,   0,   1],
        ], dtype=np.float64)

        pointsInCamera = (np.linalg.inv(world_M_camera) @ pointsInWorld.T).T
        pointsInCameraNormalized = (pointsInCamera / mvg.col(pointsInCamera[:,2]))[:,:3]
        expectedPointsInCamera = (K @ pointsInCameraNormalized.T).T

        computedPointsInCamera = mvg.project(K, np.eye(4), pointsInWorld)
        # x, y values are the same
        self.assertTrue(np.allclose(expectedPointsInCamera[:,:2], computedPointsInCamera[:,:2]))
        # z values are all 1, homogeneous
        self.assertTrue(np.allclose(computedPointsInCamera[:,2], 1))

    def teststack(self):
        A = np.array([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
        ], dtype=int)
        expectedAs = mvg.col(range(1,10))

        As = mvg.stack(A)

        self.assertTrue(np.allclose(expectedAs, As))

    def testunstack(self):
        expectedA = np.array([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
        ], dtype=int)
        As = mvg.col(range(1,10))

        A = mvg.unstack(As)

        self.assertTrue(np.allclose(expectedA, A))

    def testposeFromRT(self):
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        x = 0.1
        y = 0.5
        z = -2.0
        T = mvg.col((x, y, z))
        world_M_camera_expected = np.array([
            [0, -1, 0, x],
            [1, 0, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ])

        world_M_camera = mvg.poseFromRT(R, T)

        self.assertTrue(np.allclose(world_M_camera_expected, world_M_camera))


if __name__ == "__main__":
    unittest.main()
