import shutil
import unittest
from glob import glob
from unittest.mock import MagicMock

import numpy as np

from __context__ import src
from src import dataset


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        cls.numViews = 2
        cls.syntheticDataset = dataset.Dataset(checkerboard, virtualCamera, cls.numViews)

    def testgetCornerDetectionsInSensorCoordinates(self):
        allDetections = self.syntheticDataset.getCornerDetectionsInSensorCoordinates()

        self.assertEqual(len(allDetections), self.numViews)
        self.assertEqual(len(allDetections[0]), 2)
        self.assertEqual(allDetections[0][0].shape[1], 2)
        self.assertEqual(allDetections[0][1].shape[1], 3)

    def testwriteDatasetImages(self):
        outputFolderPath = "/tmp/output/testwritedata"
        shutil.rmtree(outputFolderPath, ignore_errors=True)

        self.syntheticDataset.writeDatasetImages(outputFolderPath)

        filePaths = glob(outputFolderPath + "/*")
        self.assertGreater(len(filePaths), 0)


if __name__ == "__main__":
    unittest.main()

