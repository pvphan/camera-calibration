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
            [1, -1, 0.4, 1],
            [-1, 1, 0.4, 1],
            [0.3, 0.1, 2.0, 1],
            [0.3, -0.1, 2.0, 1],
            [-0.8, 0.4, 1.2, 1],
            [-0.8, 0.2, 1.2, 1],
        ])

    def test_distort(self):
        k1 = 0.5
        k2 = 0.2
        p1 = 0
        p2 = 0
        k3 = 0
        distortionCoeffients = (k1, k2, p1, p2, k3)

        normalizedPointsNx2 = (self.pointsInWorld / mu.col(self.pointsInWorld[:,2]))[:,:2]
        distortedPoints = distortion.distort(normalizedPointsNx2, distortionCoeffients)

        self.assertEqual(distortedPoints.shape, normalizedPointsNx2.shape)
        self.assertEqual(normalizedPointsNx2.shape, (distortedPoints.shape[0], 2))

    def test_distortSimple(self):
        k1 = 0.5
        k2 = 0.2
        distortionCoeffients = (k1, k2)

        normalizedPointsNx2 = (self.pointsInWorld / mu.col(self.pointsInWorld[:,2]))[:,:2]
        distortedPoints = distortion.distortSimple(normalizedPointsNx2, distortionCoeffients)

        self.assertEqual(distortedPoints.shape, normalizedPointsNx2.shape)
        self.assertEqual(normalizedPointsNx2.shape, (distortedPoints.shape[0], 2))

    def test_estimateDistortion(self):
        Aexpected = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])
        width, height = 640, 480
        distortionVector = (-0.2, -0.0, 0, 0, 0)
        dataSet = dataset.createSyntheticDataset(Aexpected, width, height, distortionVector)
        dataSet.writeDatasetImages("/tmp/output/distorteddataset")
        #distortion.estimateDistortion()
        pass


if __name__ == "__main__":
    unittest.main()

