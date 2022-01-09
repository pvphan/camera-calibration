"""
Calibrates from a synthetic dataset and animates a gif of the progress
"""
import imageio

from __context__ import src
from src import calibrate
from src import dataset


class CalibrationAnimation:
    def __init__(self, syntheticDataset: dataset.Dataset):
        pass

    def writeGifs(self):
        """
        Creates a gif of the sensor point measurements being rectified
            as the estimated values of the intrinsics are updated.
            One gif per view.
        """
        pass


def createAnimation(outputFolderPath):
    width, height = 640, 480
    kExpected = (-0.5, 0.2, 0.07, -0.03, 0.05)
    syntheticDataset = dataset.createSyntheticDataset(A, width, height, kExpected)
    ani = CalibrationAnimation(syntheticDataset)
    ani.writeGifs(outputFolderPath)


def main():
    outputFolderPath = "/tmp/output/animation"
    createAnimation(outputFolderPath)


if __name__ == "__main__":
    main()
