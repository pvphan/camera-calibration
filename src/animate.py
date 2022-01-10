"""
Runs calibration and creates a gif of the reprojection error over the
course of the parameter refinement.
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
    _gifFps = 3
    _maxIters = 50
    _epsilon = 1e-12
    def __init__(self, allDetections: list, jac: jacobian.ProjectionJacobian,
            width: int, height: int):
        self._allDetections = allDetections
        self._jac = jac
        self._width = width
        self._height = height

        self._Ainitial, self._Winitial, self._kInitial = calibrate.estimateCalibrationParameters(
                self._allDetections)

    def writeGif(self, outputFolderPath):
        """
        Creates a gif of the sensor point measurements being rectified
            as the estimated values of the intrinsics are updated.
            One gif per view.
        """
        os.makedirs(outputFolderPath, exist_ok=True)
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

            imageForIteration = createProjectionErrorImage(ydot, y, self._width, self._height)
            allImages.append(imageForIteration)

            error = calibrate.computeReprojectionError(P, allDetections)
            if error < self._epsilon:
                break

            outputPath = os.path.join(outputFolderPath, f"reprojection_{i:03d}.gif")
            imageio.mimsave(outputPath, allImages, fps=self._gifFps)


def createProjectionErrorImage(ydot, y, width, height):
    length = 9
    green = (0, 255, 0)
    magenta = (255, 0, 255)
    image = visualize.createDetectionsImage(y, width, height, color=magenta)
    visualize.drawCrosses(image, ydot, length, green)
    return image


def createAnimation(outputFolderPath):
    width, height = 640, 480
    k = (-0.5, 0.2, 0.07, -0.03, 0.05)
    A = np.array([
        [415, 0, 326],
        [0, 415, 222],
        [0, 0, 1],
    ])
    syntheticDataset = dataset.createSyntheticDataset(A, width, height, k)
    allDetections = syntheticDataset.getCornerDetectionsInSensorCoordinates()
    jac = jacobian.createJacRadTan()

    ani = CalibrationAnimation(allDetections, jac, width, height)
    ani.writeGif(outputFolderPath)


def main():
    outputFolderPath = "/tmp/output/animation"
    createAnimation(outputFolderPath)


if __name__ == "__main__":
    main()
