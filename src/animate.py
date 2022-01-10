"""
Calibrates from a synthetic dataset and animates a gif of the progress
"""
import os
import imageio
import numpy as np

from __context__ import src
from src import calibrate
from src import dataset
from src import jacobian
from src import visualize


class CalibrationAnimation:
    _maxIters = 50
    def __init__(self, syntheticDataset: dataset.Dataset,
            jac: jacobian.ProjectionJacobian):
        self._syntheticDataset = syntheticDataset
        self._jac = jac

        self._allDetections = self._syntheticDataset.getCornerDetectionsInSensorCoordinates()
        self._Ainitial, self._Winitial, self._kInitial = calibrate.estimateCalibrationParameters(
                self._allDetections)

    def writeGifs(self, outputFolderPath):
        """
        Creates a gif of the sensor point measurements being rectified
            as the estimated values of the intrinsics are updated.
            One gif per view.
        """
        os.makedirs(outputFolderPath, exist_ok=True)
        width = self._syntheticDataset.getImageWidth()
        height = self._syntheticDataset.getImageHeight()
        A, W, k = self._Ainitial, self._Winitial, self._kInitial
        allDetections = self._allDetections
        allModelPoints = [modelPoints for sensorPoints, modelPoints in allDetections]
        ydot = calibrate.getSensorPoints(allDetections)
        allImages = []
        for i in range(self._maxIters):
            A, W, k = calibrate.refineCalibrationParameters(A, W, k, allDetections,
                    self._jac, maxIters=1, shouldPrint=True)
            P = calibrate.composeParameterVector(A, W, k)
            y = calibrate.projectAllPoints(P, allModelPoints)

            imageForIteration = createProjectionErrorImage(ydot, y, width, height)
            allImages.append(imageForIteration)
            outputPath = os.path.join(outputFolderPath, f"iter{i:03d}.png")
            imageio.imwrite(outputPath, imageForIteration) # TODO remove this line

            error = calibrate.computeReprojectionError(P, allDetections)
            if error < 1e-12:
                break


def createProjectionErrorImage(ydot, y, width, height):
    length = 9
    green = (0, 255, 0)
    magenta = (255, 0, 255)
    image = visualize.createDetectionsImage(y, width, height, color=magenta)
    visualize.drawCrosses(image, ydot, length, green)
    return image


def createAnimation(outputFolderPath):
    width, height = 640, 480
    kExpected = (-0.5, 0.2, 0.07, -0.03, 0.05)
    A = np.array([
        [415, 0, 326],
        [0, 415, 222],
        [0, 0, 1],
    ])
    syntheticDataset = dataset.createSyntheticDataset(A, width, height, kExpected)
    jac = jacobian.createJacRadTan()

    ani = CalibrationAnimation(syntheticDataset, jac)
    ani.writeGifs(outputFolderPath)


def main():
    outputFolderPath = "/tmp/output/animation"
    createAnimation(outputFolderPath)


if __name__ == "__main__":
    main()
