import unittest

import numpy as np

from __context__ import src
from src import noise


class TestNoise(unittest.TestCase):
    def test_applyNoise(self):
        standardDeviation = 0.5
        noiseModel = noise.NoiseModel(standardDeviation)
        imageHeight, imageWidth = (480, 640)
        sensorPoints = np.random.uniform(0, 100, size=(imageHeight,imageWidth))

        noisyPoints = noiseModel.applyNoise(sensorPoints)

        self.assertEqual(sensorPoints.shape, noisyPoints.shape)
        self.assertFalse(np.allclose(sensorPoints, noisyPoints))


if __name__ == "__main__":
    unittest.main()

