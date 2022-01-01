import unittest
from unittest.mock import MagicMock

import numpy as np

from __context__ import src
from src import dataset


class TestDataset(unittest.TestCase):
    def testgetCornerDetectionsInSensorCoordinates(self):
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

        virtualCamera = MagicMock()
        measuredPointsInSensor = np.array([
            [100.0, 200.0],
            [300.0, 400.0],
        ])
        measuredPointsInBoard = np.array([
            [0.100, 0.200, 0],
            [0.300, 0.400, 0],
        ])
        virtualCamera.measureDetectedPoints.return_value = (measuredPointsInSensor,
                measuredPointsInBoard)
        numViews = 2
        syntheticDataset = dataset.Dataset(checkerboard, virtualCamera, numViews)

        allDetections = syntheticDataset.getCornerDetectionsInSensorCoordinates()

        self.assertEqual(len(allDetections), numViews)
        self.assertEqual(len(allDetections[0]), 2)
        self.assertEqual(allDetections[0][0].shape[1], 2)
        self.assertEqual(allDetections[0][1].shape[1], 3)


if __name__ == "__main__":
    unittest.main()

