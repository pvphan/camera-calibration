import unittest

import numpy as np

from calibration import mathutils as mu


class TestMathUtils(unittest.TestCase):
    def testcol(self):
        v = mu.col((1,1,1))
        self.assertEqual(v.shape, (3,1))

    def testskew(self):
        v = mu.col((0.5, 1.0, -0.25))
        vHatExpected = np.array([
            [0, 0.25, 1.0],
            [-0.25, 0, -0.5],
            [-1.0, 0.5, 0],
        ])

        vHat = mu.skew(v)

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

        v = mu.unskew(vHat)
        self.assertTrue(np.allclose(vExpected, v))

    def testexp(self):
        z = mu.col((0, 0, 1))
        theta = 90
        expectedRotation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        computedRotation = mu.exp(np.radians(theta) * mu.skew(z))
        self.assertEqual(computedRotation.shape, (3,3))
        self.assertTrue(np.allclose(expectedRotation, computedRotation))

    def testexpZeroRot(self):
        z = mu.col((0, 0, 1))
        theta = 0
        expectedRotation = np.eye(3)
        computedRotation = mu.exp(np.radians(theta) * mu.skew(z))

        self.assertEqual(computedRotation.shape, (3,3))
        self.assertTrue(np.allclose(expectedRotation, computedRotation))

    def testeulerToRotationMatrix1(self):
        expectedRotation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        eulerAngles = (0, 0, 90)

        computedRotation = mu.eulerToRotationMatrix(eulerAngles)

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

        computedRotation = mu.eulerToRotationMatrix(eulerAngles)

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

        computedRotation = mu.eulerToRotationMatrix(eulerAngles)

        self.assertEqual(computedRotation.shape, (3,3))
        self.assertTrue(np.allclose(expectedRotation, computedRotation),
                f"\nexpected:\n{expectedRotation}\n\ncomputed:\n{computedRotation}")

    def teststack(self):
        A = np.array([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
        ], dtype=int)
        expectedAs = mu.col(range(1,10))

        As = mu.stack(A)

        self.assertTrue(np.allclose(expectedAs, As))

    def testunstack(self):
        expectedA = np.array([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
        ], dtype=int)
        As = mu.col(range(1,10))

        A = mu.unstack(As)

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
        T = mu.col((x, y, z))
        world_M_camera_expected = np.array([
            [0, -1, 0, x],
            [1, 0, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ])

        world_M_camera = mu.poseFromRT(R, T)

        self.assertTrue(np.allclose(world_M_camera_expected, world_M_camera))


if __name__ == "__main__":
    unittest.main()
