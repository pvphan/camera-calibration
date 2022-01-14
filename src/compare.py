from glob import glob

import numpy as np

from __context__ import src
from src import main as mainmodule


def readCacheFile(detectionsCachePath):
    detectionsData = np.load(detectionsCachePath, allow_pickle=True)
    sensorPoints = detectionsData[()]["uvs"].reshape(-1, 2)
    ids = detectionsData[()]["ids"].ravel()

    modelPoints = np.empty((0,3))

    # checkerSize = 0.030
    numMarkersHeight = 18
    numMarkersWidth = 25
    boardPhysicalHeightMeters = 0.540
    boardPhysicalWidthMeters = 0.750

    spacing = boardPhysicalHeightMeters / numMarkersHeight
    for id in ids:
        row = id // numMarkersWidth
        col = id % numMarkersWidth
        X = col * spacing
        Y = row * spacing
        Z = 0.0
        modelPoint = np.array(((X, Y, Z)))
        modelPoints = np.vstack((modelPoints, modelPoint))

    return sensorPoints, modelPoints


def main():
    detectionsCachePaths = sorted(glob("/tmp/output/detectioncache1/*_left.npy"))
    allDetections = []
    for detectionsCachePath in detectionsCachePaths:
        detections = readCacheFile(detectionsCachePath)
        allDetections.append(detections)
    sse, Afinal, Wfinal, kFinal = mainmodule.calibrateCamera(allDetections[:10], "radtan")


if __name__ == "__main__":
    main()

