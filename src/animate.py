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
from src import distortion
from src import visualize


class CalibrationAnimation:
    _gifFps = 5
    _maxIters = 50
    _epsilon = 1e-5
    def __init__(self, calibrator: calibrate.Calibrator, allDetections: list, width: int, height: int):
        self._calibrator = calibrator
        self._allDetections = allDetections
        self._width = width
        self._height = height

    def writeGif(self, outputFolderPath):
        """
        Creates a gif of the projected model points and sensor points
            as the estimated values of the intrinsics and extrinsics are updated.
        """
        os.makedirs(outputFolderPath, exist_ok=True)
        A, W, k = self._calibrator.estimateCalibrationParameters(
                self._allDetections)
        allDetections = self._allDetections
        allModelPoints = [modelPoints for sensorPoints, modelPoints in allDetections]
        ydot = calibrate.getSensorPoints(allDetections)
        allImages = []
        for i in range(self._maxIters):
            sse, A, W, k = self._calibrator.refineCalibrationParameters(A, W, k, allDetections,
                    maxIters=1, shouldPrint=True)
            P = self._calibrator._composeParameterVector(A, W, k)
            y = self._calibrator.projectAllPoints(P, allModelPoints)

            imageForIteration = createProjectionErrorImage(ydot, y, self._width, self._height)
            allImages.append(imageForIteration)

            if sse < self._epsilon:
                break

        outputPath = os.path.join(outputFolderPath, f"reprojection.gif")
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
    A = np.array([
        [415, 0, 326],
        [0, 415, 222],
        [0, 0, 1],
    ])
    isFisheye = False
    if isFisheye:
        k = (-0.5, 0.2, 0.07, -0.03)
        syntheticDataset = dataset.createSyntheticDatasetFisheye(A, width, height, k)
    else:
        k = (-0.5, 0.2, 0.07, -0.03, 0.05)
        syntheticDataset = dataset.createSyntheticDatasetRadTan(A, width, height, k)
    allDetections = syntheticDataset.getCornerDetectionsInSensorCoordinates()
    distortionModel = distortion.RadialTangentialModel()
    calibrator = calibrate.Calibrator(distortionModel)

    ani = CalibrationAnimation(calibrator, allDetections, width, height)
    ani.writeGif(outputFolderPath)


def main():
    outputFolderPath = "/tmp/output/animation"
    createAnimation(outputFolderPath)


if __name__ == "__main__":
    main()
