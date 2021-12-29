"""
Generates synthetic datasets to test calibration code.
"""

import numpy as np


class Checkerboard:
    numCornersWidth = 9
    numCornersHeight = 6
    spacing = 0.050

    def getCornerPositions(self):
        pass


class Dataset:
    def __init__(self, checkerboard: Checkerboard, K: np.ndarray):
        pass

