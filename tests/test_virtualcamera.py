import unittest
from unittest.mock import MagicMock

import numpy as np

from __context__ import src
from src import virtualcamera
from src import mathutils as mu


class TestVirtualCamera(unittest.TestCase):
    def test_measureDetectedPoints(self):
        imageWidth = 640
        imageHeight = 480
        intrinsicMatrix = np.array([
            [450,   0, 360],
            [  0, 450, 240],
            [  0,   0,   1],
        ], dtype=np.float64)
        k1 = 0.5
        k2 = 0.2
        distortionCoeffients = (k1, k2)
        checkerboard = MagicMock()
        cornerPositions = np.array([
            [0.0, 0.0, 0],
            [0.1, 0.0, 0],
            [0.2, 0.0, 0],
            [0.3, 0.0, 0],
            [0.0, 0.1, 0],
            [0.1, 0.1, 0],
            [0.2, 0.1, 0],
            [0.3, 0.1, 0],
        ])
        checkerboard.getCornerPositions.return_value = cornerPositions

        virtualCamera = virtualcamera.VirtualCamera(intrinsicMatrix, distortionCoeffients,
                imageWidth, imageHeight)

        R = mu.eulerToRotationMatrix((0, 0, 180))
        t = (0.05, 0.05, 2)
        cameraPoseInBoard = mu.poseFromRT(R, t)
        boardPoseInCamera = np.linalg.inv(cameraPoseInBoard)

        measuredPointsInSensor, measuredPointsInBoard = virtualCamera.measureDetectedPoints(
                checkerboard, boardPoseInCamera)

        self.assertEqual(measuredPointsInSensor.shape[0], measuredPointsInBoard.shape[0])
        self.assertEqual(measuredPointsInSensor.shape[1], 2)
        self.assertEqual(measuredPointsInBoard.shape[1], 3)


if __name__ == "__main__":
    unittest.main()

