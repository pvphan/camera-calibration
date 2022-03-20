import os
from glob import glob

import numpy as np

from __context__ import src
from src import checkerboard
from src import main as mainmodule
from src import visualize


def readCacheFile(checkerBoard, detectionsCachePath):
    detectionsData = np.load(detectionsCachePath, allow_pickle=True)
    sensorPoints = detectionsData[()]["uvs"].reshape(-1, 2)
    ids = detectionsData[()]["ids"].ravel()
    modelPoints = checkerBoard.getCornerPositions(ids)
    return sensorPoints.astype(np.float64), modelPoints.astype(np.float64)


def main():
    shouldVisualize = False
    detectionsCachePaths = sorted(glob("/tmp/output/dataset1/detectioncache/*_left.npy"))
    outputFolderPath = "/tmp/output/test1"
    numCornersWidth = 25
    numCornersHeight = 18
    spacing = 0.030
    width, height = 1440, 1080
    checkerBoard = checkerboard.Checkerboard(numCornersWidth, numCornersHeight, spacing)

    os.makedirs(outputFolderPath, exist_ok=True)
    allDetections = []
    for detectionsCachePath in detectionsCachePaths:
        detections = readCacheFile(checkerBoard, detectionsCachePath)
        allDetections.append(detections)

        if shouldVisualize:
            fileName = os.path.basename(detectionsCachePath)
            outputFilePath = os.path.join(outputFolderPath, fileName.replace(".npy", ".png"))
            sensorPoints, modelPoints = detections
            visualize.writeDetectionsImage(sensorPoints, width, height, outputFilePath)
    maxIters = 100
    sse, Afinal, Wfinal, kFinal = mainmodule.calibrateCamera(allDetections, "radtan", maxIters)


if __name__ == "__main__":
    main()

