import numpy as np


class NoiseModel:
    def __init__(self, standardDeviation: float):
        self._standardDeviation = standardDeviation

    def applyNoise(self, sensorPoints: np.ndarray):
        """
        Input:
            measurements -- (N,2) array of sensor points

        Output:
            sensorPointsWithNoise -- (N,2) array of sensor
                points after noise is applied
        """
        noise = np.random.normal(0.0, self._standardDeviation, sensorPoints.shape)
        return sensorPoints + noise
