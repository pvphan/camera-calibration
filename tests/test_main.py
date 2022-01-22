import unittest

import numpy as np

from __context__ import src
from src import main
from src import noise
from src import dataset


class TestMain(unittest.TestCase):
    def setUp(self):
        self.width, self.height = 640, 480
        self.Aexpected = np.array([
            [395.1, 0, 320.5],
            [0, 405.8, 249.2],
            [0, 0, 1],
        ])
        self.kRadTan = (-0.5, 0.2, 0.07, -0.03, 0.05)

    def test_calibrateCamera(self):
        width, height = self.width, self.height
        Aexpected = self.Aexpected
        kExpected = self.kRadTan
        noiseModel = None
        syntheticDataset = dataset.createSyntheticDatasetRadTan(
                Aexpected, width, height, kExpected, noiseModel)
        Wexpected = syntheticDataset.getAllBoardPosesInCamera()
        allDetections = syntheticDataset.getCornerDetectionsInSensorCoordinates()
        distortionType = "radtan"
        maxIters = 100

        sse, Acomputed, Wcomputed, kComputed = main.calibrateCamera(
                allDetections, distortionType, maxIters)

        self.assertAlmostEqual(sse, 0)
        self.assertAllClose(Aexpected, Acomputed)
        self.assertAllClose(Wexpected, Wcomputed)
        self.assertAllClose(kExpected, kComputed)

    def test_calibrateCamera2(self):
        realisticDataset = dataset.createRealisticRadTanDataset()
        Aexpected = realisticDataset.getIntrinsicMatrix()
        Wexpected = realisticDataset.getAllBoardPosesInCamera()
        kExpected = realisticDataset.getDistortionVector()
        allDetections = realisticDataset.getCornerDetectionsInSensorCoordinates()
        distortionType = "radtan"
        maxIters = 100

        sse, Acomputed, Wcomputed, kComputed = main.calibrateCamera(
                allDetections, distortionType, maxIters)

        self.assertAlmostEqual(sse, 0)
        self.assertAllClose(Aexpected, Acomputed)
        self.assertAllClose(Wexpected, Wcomputed)
        self.assertAllClose(kExpected, kComputed)

    def test_calibrateCameraWithNoise(self):
        width, height = self.width, self.height
        Aexpected = self.Aexpected
        kExpected = self.kRadTan
        noiseModel = noise.NoiseModel(0.1)
        syntheticDataset = dataset.createSyntheticDatasetRadTan(
                Aexpected, width, height, kExpected, noiseModel)
        Wexpected = syntheticDataset.getAllBoardPosesInCamera()
        allDetections = syntheticDataset.getCornerDetectionsInSensorCoordinates()
        distortionType = "radtan"
        maxIters = 100

        sse, Acomputed, Wcomputed, kComputed = main.calibrateCamera(
                allDetections, distortionType, maxIters)

        self.assertAllClose(Aexpected, Acomputed, atol=2.0)
        self.assertAllClose(kExpected, kComputed, atol=0.05)

    def test_calibrateCameraFisheye(self):
        width, height = self.width, self.height
        Aexpected = self.Aexpected
        kExpected = (-0.155, -0.02, 0.0, -0.03)
        noiseModel = None
        syntheticDataset = dataset.createSyntheticDatasetFisheye(
                Aexpected, width, height, kExpected, noiseModel)
        syntheticDataset.writeDatasetImages("/tmp/output/test_calibrateCameraFisheye")
        Wexpected = syntheticDataset.getAllBoardPosesInCamera()
        allDetections = syntheticDataset.getCornerDetectionsInSensorCoordinates()
        distortionType = "fisheye"
        maxIters = 100

        sse, Acomputed, Wcomputed, kComputed = main.calibrateCamera(
                allDetections, distortionType, maxIters)

        self.assertAlmostEqual(sse, 0)
        self.assertAllClose(Aexpected, Acomputed)
        self.assertAllClose(Wexpected, Wcomputed)
        self.assertAllClose(kExpected, kComputed)

    def assertAllClose(self, A, B, atol=1e-9):
        self.assertTrue(np.allclose(A, B, atol=atol),
                f"\n{A} \n != \n {B}")


if __name__ == "__main__":
    unittest.main()
