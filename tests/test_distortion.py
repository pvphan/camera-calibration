import unittest

import numpy as np

from __context__ import src
from src import dataset
from src import distortion
from src import mathutils as mu


class TestCalibrate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pointsInWorld = np.array([
            [1, -1, 0.4],
            [-1, 1, 0.4],
            [0.3, 0.1, 2.0],
            [0.3, -0.1, 2.0],
            [-0.8, 0.4, 1.2],
            [-0.8, 0.2, 1.2],
        ])

    def test_distortPoints(self):
        k1 = 0.5
        k2 = 0.2
        p1 = 0
        p2 = 0
        k3 = 0
        distortionCoeffients = (k1, k2, p1, p2, k3)

        normalizedPointsNx2 = mu.projectStandard(self.pointsInWorld)
        distortedPoints = distortion.distortPoints(normalizedPointsNx2, distortionCoeffients)

        self.assertEqual(distortedPoints.shape, normalizedPointsNx2.shape)
        self.assertEqual(normalizedPointsNx2.shape, (distortedPoints.shape[0], 2))
        self.assertFalse(np.allclose(normalizedPointsNx2, distortedPoints))

    def test_estimateDistortion(self):
        A = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])
        width, height = 640, 480
        kExpected = (-0.5, 0.2)
        dataSet = dataset.createSyntheticDataset(A, width, height, kExpected)
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()

        kComputed = distortion.estimateDistortion(A, allDetections)

        self.assertTrue(np.allclose(kExpected, kComputed))


if __name__ == "__main__":
    unittest.main()

