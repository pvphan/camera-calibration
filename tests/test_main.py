import unittest

import numpy as np

from __context__ import src
from src import main
from src import dataset


class TestMain(unittest.TestCase):
    def test_calibrateCamera(self):
        width, height = 640, 480
        Aexpected = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])
        kExpected = (-0.5, 0.2, 0.07, -0.03, 0.05)
        syntheticDataset = dataset.createSyntheticDataset(Aexpected, width, height, kExpected)
        allDetections = syntheticDataset.getCornerDetectionsInSensorCoordinates()

        Acomputed, kComputed = main.calibrateCamera(allDetections)
        self.assertAllClose(Aexpected, Acomputed)
        self.assertAllClose(kExpected, kComputed)

    def assertAllClose(self, A, B, atol=1e-9):
        self.assertTrue(np.allclose(A, B, atol=atol),
                f"\n{A} \n != \n {B}")


if __name__ == "__main__":
    unittest.main()
