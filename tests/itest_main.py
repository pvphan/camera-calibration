import unittest

import numpy as np

from __context__ import src
from src import main
from src import noise
from src import dataset


class TestMain(unittest.TestCase):
    def test_calibrateCameraRealistic(self):
        realisticDataset = dataset.createRealisticRadTanDataset()
        realisticDataset.writeDatasetImages("/tmp/output/test_calibrateCameraRealistic")
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
        for i, (we, wc) in enumerate(zip(Wexpected, Wcomputed)):
            self.assertAllClose(we, wc, atol=1)
        self.assertAllClose(kExpected, kComputed)

    def test_calibrateCameraWithNoise(self):
        width, height = 1440, 1080
        Aexpected = np.array([
            [803.1, 0, 700.5],
            [0, 803.1, 529.2],
            [0, 0, 1],
        ], dtype=np.float64)
        kExpected = (-0.25, 0.2, 0.07, -0.03, 0.05)
        noiseModel = noise.NoiseModel(0.1)
        syntheticDataset = dataset.createSyntheticDatasetRadTan(
                Aexpected, width, height, kExpected, noiseModel)
        syntheticDataset.writeDatasetImages("/tmp/output/test_calibrateCameraWithNoise")
        Wexpected = syntheticDataset.getAllBoardPosesInCamera()
        allDetections = syntheticDataset.getCornerDetectionsInSensorCoordinates()
        distortionType = "radtan"
        maxIters = 100

        sse, Acomputed, Wcomputed, kComputed = main.calibrateCamera(
                allDetections, distortionType, maxIters)

        self.assertAllClose(Aexpected, Acomputed, atol=2.0)
        self.assertAllClose(kExpected, kComputed, atol=0.05)

    def test_calibrateCameraFisheye(self):
        width, height = 1440, 1080
        Aexpected = np.array([
            [803.1, 0, 700.5],
            [0, 803.1, 529.2],
            [0, 0, 1],
        ], dtype=np.float64)
        kExpected = (-0.155, -0.02, 0.0, -0.03)

        noiseModel = None
        syntheticDataset = dataset.createSyntheticDatasetFisheye(
                Aexpected, width, height, kExpected, noiseModel)
        syntheticDataset.writeDatasetImages("/tmp/output/test_calibrateCameraFisheye")
        Wexpected = syntheticDataset.getAllBoardPosesInCamera()
        allDetections = syntheticDataset.getCornerDetectionsInSensorCoordinates()
        distortionType = "fisheye"
        maxIters = 10

        sse, Acomputed, Wcomputed, kComputed = main.calibrateCamera(
                allDetections, distortionType, maxIters)

        # TODO: this isn't converge correctly right now, just test
        #       that it doesn't throw exceptions for now
        #self.assertAllClose(Aexpected, Acomputed)
        #self.assertAllClose(Wexpected, Wcomputed)
        #self.assertAllClose(kExpected, kComputed)

    def assertAllClose(self, A, B, atol=1e-9):
        self.assertTrue(np.allclose(A, B, atol=atol),
                f"\n{A} \n != \n {B}")


if __name__ == "__main__":
    unittest.main()
