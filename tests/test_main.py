import unittest

import numpy as np

from __context__ import src
from src import main
from src import dataset


class TestMain(unittest.TestCase):
    def test_calibrateCamera(self):
        width, height = 640, 480
        Aexpected = np.array([
            [395.1, 0, 320.5],
            [0, 405.8, 249.2],
            [0, 0, 1],
        ])
        kExpected = (-0.5, 0.2, 0.07, -0.03, 0.05)
        syntheticDataset = dataset.createSyntheticDatasetRadTan(Aexpected, width, height, kExpected)
        Wexpected = syntheticDataset.getAllBoardPosesInCamera()
        allDetections = syntheticDataset.getCornerDetectionsInSensorCoordinates()
        distortionType = "radtan"

        sse, Acomputed, Wcomputed, kComputed = main.calibrateCamera(allDetections, distortionType)

        self.assertAlmostEqual(sse, 0)
        self.assertAllClose(Aexpected, Acomputed)
        self.assertAllClose(Wexpected, Wcomputed)
        self.assertAllClose(kExpected, kComputed)

    def test_calibrateCameraFisheye(self):
        width, height = 640, 480
        Aexpected = np.array([
            [395.1, 0, 310.5],
            [0, 405.8, 249.2],
            [0, 0, 1],
        ])
        kExpected = (0.5, 0.2, 0.07, -0.03)
        syntheticDataset = dataset.createSyntheticDatasetFisheye(Aexpected, width, height, kExpected)
        Wexpected = syntheticDataset.getAllBoardPosesInCamera()
        allDetections = syntheticDataset.getCornerDetectionsInSensorCoordinates()
        distortionType = "fisheye"

        sse, Acomputed, Wcomputed, kComputed = main.calibrateCamera(allDetections, distortionType)

        self.assertAlmostEqual(sse, 0)
        self.assertAllClose(Aexpected, Acomputed)
        self.assertAllClose(Wexpected, Wcomputed)
        self.assertAllClose(kExpected, kComputed)

    def assertAllClose(self, A, B, atol=1e-9):
        self.assertTrue(np.allclose(A, B, atol=atol),
                f"\n{A} \n != \n {B}")


if __name__ == "__main__":
    unittest.main()
