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
    return sensorPoints, modelPoints


def main():
    shouldVisualize = False
    detectionsCachePaths = sorted(glob("/tmp/output/detectioncache0/*_right.npy"))
    outputFolderPath = "/tmp/output/test0"
    numCornersWidth = 25
    numCornersHeight = 18
    spacing = 0.030
    width, height = 1440, 1080
    checkerBoard = checkerboard.Checkerboard(numCornersWidth, numCornersHeight, spacing)

    os.makedirs(outputFolderPath, exist_ok=True)
    allDetections = []
    for detectionsCachePath in detectionsCachePaths:
        fileName = os.path.basename(detectionsCachePath)
        outputFilePath = os.path.join(outputFolderPath, fileName.replace(".npy", ".png"))
        detections = readCacheFile(checkerBoard, detectionsCachePath)
        sensorPoints, modelPoints = detections

        if shouldVisualize:
            visualize.writeDetectionsImage(sensorPoints, width, height, outputFilePath)

        allDetections.append(detections)
    sse, Afinal, Wfinal, kFinal = mainmodule.calibrateCamera(allDetections, "radtan")


if __name__ == "__main__":
    main()

