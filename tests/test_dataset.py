import os
import shutil
import unittest
from glob import glob
from unittest.mock import MagicMock

import imageio
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
        measuredIds = (0, 1)
        measuredPointsInSensor = np.array([
            [100.0, 200.0],
            [300.0, 400.0],
        ])
        measuredPointsInBoard = np.array([
            [0.100, 0.200, 0],
            [0.300, 0.400, 0],
        ])
        virtualCamera.measureBoardPoints.return_value = (
                measuredIds, measuredPointsInSensor, measuredPointsInBoard)
        cls.imageWidth = 640
        cls.imageHeight = 480
        virtualCamera.getImageWidth.return_value = cls.imageWidth
        virtualCamera.getImageHeight.return_value = cls.imageHeight
        cls.numViews = 2
        cls.syntheticDataset = dataset.Dataset(checkerboard, virtualCamera, cls.numViews)
        cls.outputFolderPath = "/tmp/output/testwritedata"

    def test_getCornerDetectionsInSensorCoordinates(self):
        allDetections = self.syntheticDataset.getCornerDetectionsInSensorCoordinates()

        self.assertEqual(len(allDetections), self.numViews)
        self.assertEqual(len(allDetections[0]), 2)
        self.assertEqual(allDetections[0][0].shape[1], 2)
        self.assertEqual(allDetections[0][1].shape[1], 3)

    def test_getAllBoardPosesInCamera(self):
        allBoardPosesInCamera = self.syntheticDataset.getAllBoardPosesInCamera()

        self.assertEqual(len(allBoardPosesInCamera), self.numViews)
        self.assertEqual(allBoardPosesInCamera[0].shape, (4,4))

    def test_writeDatasetImages(self):
        shutil.rmtree(self.outputFolderPath, ignore_errors=True)
        self.syntheticDataset.writeDatasetImages(self.outputFolderPath)

        filePaths = glob(self.outputFolderPath + "/*")
        self.assertGreater(len(filePaths), 0)
        expectedImageShape = (self.imageHeight, self.imageWidth, 3)
        for filePath in filePaths:
            image = imageio.imread(filePath)
            self.assertEqual(image.shape, expectedImageShape)

    def test_exportDetections(self):
        shutil.rmtree(self.outputFolderPath, ignore_errors=True)
        os.makedirs(self.outputFolderPath, exist_ok=True)
        outputFilePath = os.path.join(self.outputFolderPath, "detections.json")
        allDetectionsExpected = self.syntheticDataset.getCornerDetectionsInSensorCoordinates()

        self.syntheticDataset.exportDetections(outputFilePath)

        self.assertTrue(os.path.exists(outputFilePath))

        allDetections = dataset.createDetectionsFromPath(outputFilePath)

        self.assertEqual(len(allDetections), self.numViews)
        self.assertEqual(len(allDetections[0]), 2)
        self.assertEqual(allDetections[0][0].shape[1], 2)
        self.assertEqual(allDetections[0][1].shape[1], 3)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.outputFolderPath, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

