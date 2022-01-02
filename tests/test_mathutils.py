import unittest

import numpy as np

from __context__ import src
from src import mathutils as mu


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

    def testtransformRepresentations(self):
        # preferred representation: pose of camera in world <--> transform from camera to world
        eulerAngles = (0, 45, 90)
        rotationMatrix = mu.eulerToRotationMatrix(eulerAngles)
        translationVector = (0.4, 0.1, -0.7)
        world_M_camera = mu.poseFromRT(rotationMatrix, translationVector)

        pointInWorld = (0, 1.2, -1.0)
        pointInWorldHomog = mu.homog(pointInWorld)
        camera_M_world = np.linalg.inv(world_M_camera)
        pointInCamera1 = (camera_M_world @ pointInWorldHomog.T).T

        # [R t] representation of extrinsics from literature
        #   essentially the inverse transform and baking in the standard projection matrix
        RT = np.zeros((3,4))
        RT[:3,:3] = rotationMatrix.T
        RT[:3,3] = -rotationMatrix.T @ translationVector
        pointInCamera2 = (RT @ pointInWorldHomog.T).T

        self.assertTrue(np.allclose(pointInCamera1[0,:3], pointInCamera2[0,:3]))
        RTHomog = np.eye(4)
        RTHomog[:3,:] = RT
        self.assertTrue(np.allclose(camera_M_world, RTHomog))

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

    def testproject(self):
        world_M_camera = np.eye(4)
        pointsInWorld = np.array([
            [1, -1, 0.4, 1],
            [-1, 1, 0.4, 1],
            [0.3, 0.1, 2.0, 1],
            [0.3, -0.1, 2.0, 1],
            [-0.8, 0.4, 1.2, 1],
            [-0.8, 0.2, 1.2, 1],
        ])
        A = np.array([
            [450,   0, 360],
            [  0, 450, 240],
            [  0,   0,   1],
        ], dtype=np.float64)
        pointsInCamera = (np.linalg.inv(world_M_camera) @ pointsInWorld.T).T
        pointsInCameraNormalized = (pointsInCamera / mu.col(pointsInCamera[:,2]))[:,:3]
        expectedPointsInCamera = (A @ pointsInCameraNormalized.T).T

        computedPointsInCamera = mu.project(A, np.eye(4), pointsInWorld)
        # x, y values are the same
        self.assertTrue(np.allclose(expectedPointsInCamera[:,:2], computedPointsInCamera[:,:2]))
        # z values are all 1, homogeneous
        self.assertTrue(np.allclose(computedPointsInCamera[:,2], 1))


if __name__ == "__main__":
    unittest.main()
